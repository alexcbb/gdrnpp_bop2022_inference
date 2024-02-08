import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)


from predictor_yolo import YoloPredictor
from predictor_gdrn import GdrnPredictor
import os
import datetime

import numpy as np

import cv2
import torch 
import json
import numpy as np
from lib.pysixd import inout
from core.utils.data_utils import crop_resize_by_warp_affine, get_2d_coord_np
from detectron2.structures import BoxMode
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
from lib.utils.mask_utils import get_edge
import mmcv

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
def get_image_list(rgb_images_path, depth_images_path=None):
    image_names = []

    rgb_file_names = os.listdir(rgb_images_path)
    rgb_file_names.sort()
    for filename in rgb_file_names:
        apath = os.path.join(rgb_images_path, filename)
        ext = os.path.splitext(apath)[1]
        if ext in IMAGE_EXT:
            image_names.append(apath)

    if depth_images_path is not None:
        depth_file_names = os.listdir(depth_images_path)
        depth_file_names.sort()
        for i, filename in enumerate(depth_file_names):
            apath = os.path.join(depth_images_path, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names[i] = (image_names[i], apath)
                # depth_names.append(apath)

    else:
        for i, filename in enumerate(rgb_file_names):
            image_names[i] = (image_names[i], None)

    return image_names

def preprocessing(outputs, image, extents, cam):
    """
    Preprocessing detection model output and input image
    Args:
        outputs: yolo model output
        image: rgb image
    Returns:
        dict
    """

    dataset_dict = {
        "cam": cam,
        "annotations": []
    }
    boxes = outputs[0]

    # print(f"Number of boxes: {len(boxes)}")
    for i in range(len(boxes)):
        annot_inst = {}
        box = boxes[i].tolist()
        annot_inst["category_id"] = int(box[6])
        annot_inst["score"] = box[4] * box[5]
        annot_inst["bbox_est"] = [box[0], box[1], box[2], box[3]]
        annot_inst["bbox_mode"] = BoxMode.XYXY_ABS
        dataset_dict["annotations"].append(annot_inst)


    # other transforms (mainly geometric ones);
    # for 6d pose task, flip is not allowed in general except for some 2d keypoints methods
    im_H, im_W = image.shape[:2]  # h, w

    input_res = 256
    out_res = 64
    # CHW -> HWC
    coord_2d = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)

    # here get batched rois
    roi_infos = {}
    # yapf: disable
    roi_keys = ["scene_im_id", "file_name", "cam", "im_H", "im_W",
                "roi_img", "inst_id", "roi_coord_2d", "roi_coord_2d_rel",
                "roi_cls", "score", "time", "roi_extent",
                "bbox_est", "bbox_mode", "bbox_center", "roi_wh",
                "scale", "resize_ratio", "model_info",
                ]

    for _key in roi_keys:
        roi_infos[_key] = []
    for inst_i, inst_infos in enumerate(dataset_dict["annotations"]):
        # inherent image-level infos
        roi_infos["im_H"].append(im_H)
        roi_infos["im_W"].append(im_W)
        roi_infos["cam"].append(dataset_dict["cam"])

        # roi-level infos
        roi_infos["inst_id"].append(inst_i)
        # roi_infos["model_info"].append(inst_infos["model_info"])

        roi_cls = inst_infos["category_id"]
        roi_infos["roi_cls"].append(roi_cls)
        roi_infos["score"].append(inst_infos.get("score", 1.0))

        roi_infos["time"].append(inst_infos.get("time", 0))
        
        # extent
        roi_extent = extents[roi_cls]
        roi_infos["roi_extent"].append(roi_extent)

        # TODO: adjust amodal bbox here
        bbox = np.array(inst_infos["bbox_est"])
        roi_infos["bbox_est"].append(bbox)
        roi_infos["bbox_mode"].append(BoxMode.XYXY_ABS)
        x1, y1, x2, y2 = bbox
        bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        scale = max(bh, bw) * 1.5 #cfg.INPUT.DZI_PAD_SCALE
        scale = min(scale, max(im_H, im_W)) * 1.0

        roi_infos["bbox_center"].append(bbox_center.astype("float32"))
        roi_infos["scale"].append(scale)
        roi_wh = np.array([bw, bh], dtype=np.float32)
        roi_infos["roi_wh"].append(roi_wh)
        roi_infos["resize_ratio"].append(out_res / scale)

        # CHW, float32 tensor
        # roi_image
        roi_img = crop_resize_by_warp_affine(
            image, bbox_center, scale, input_res, interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)

        roi_img = roi_img / np.array([255.0, 255.0, 255.0]).reshape(-1, 1, 1)
        roi_infos["roi_img"].append(roi_img.astype("float32"))

        # roi_coord_2d
        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
        ).transpose(
            2, 0, 1
        )  # HWC -> CHW
        roi_infos["roi_coord_2d"].append(roi_coord_2d.astype("float32"))

    for _key in roi_keys:
        if _key in ["roi_img", "roi_coord_2d", "roi_coord_2d_rel", "roi_depth"]:
            dataset_dict[_key] = torch.as_tensor(np.array(roi_infos[_key])).contiguous()
        elif _key in ["model_info", "scene_im_id", "file_name"]:
            # can not convert to tensor
            dataset_dict[_key] = roi_infos[_key]
        else:
            if isinstance(roi_infos[_key], list):
                dataset_dict[_key] = torch.as_tensor(np.array(roi_infos[_key]))
            else:
                dataset_dict[_key] = torch.as_tensor(roi_infos[_key])

    roi_keys = ["im_H", "im_W",
                "roi_img", "inst_id", "roi_coord_2d", "roi_coord_2d_rel",
                "roi_cls", "score", "time", "roi_extent",
                "bbox", "bbox_est", "bbox_mode", "roi_wh",
                "scale", "resize_ratio",
                ]
    batch = {}
    roi_cls = []
    for key in roi_keys:
        if key in ["roi_cls"]:
            dtype = torch.long
        else:
            dtype = torch.float32
        if key in dataset_dict:
            batch[key] = torch.cat([dataset_dict[key]], dim=0).to(device='cuda', dtype=dtype)

    batch["roi_cam"] = torch.cat([dataset_dict["cam"]], dim=0).to('cuda', non_blocking=True)
    batch["roi_center"] = torch.cat([dataset_dict["bbox_center"]], dim=0).to('cuda', non_blocking=True)
    batch["bbox_center"] = torch.cat([dataset_dict["bbox_center"]], dim=0).to('cuda', non_blocking=True)#roi_infos["bbox_center"]
    batch["cam"] = dataset_dict["cam"]

    return batch

def postprocessing(out_dict, objs, outputs):
    """
    Postprocess the gdrn model outputs
    Args:
        data_dict: gdrn model preprocessed data
        out_dict: gdrn model output
    Returns:
        dict: poses of objects
    """
    boxes = outputs[0]
    data_dict = {
        "cur_res": [],
    }
    i_out = -1
    for i_inst in range(len(boxes)):
        box = boxes[i_inst].tolist()

        i_out += 1

        cur_res = {
            "obj_id": objs[int(box[6])+1],
            "score": box[4] * box[5],
            "bbox_est": [box[0], box[1], box[2], box[3]],  # xyxy
        }
        #if cfg.TEST.USE_PNP:
        #    pose_est_pnp = get_pnp_ransac_pose(cfg, data_dict, out_dict, i_inst, i_out)
        #    cur_res["R"] = pose_est_pnp[:3, :3]
        #    cur_res["t"] = pose_est_pnp[:3, 3]
        #else:
        cur_res.update(
            {
                "R": out_dict["rot"][i_out].detach() .cpu().numpy(),
                "t": out_dict["trans"][i_out].detach().cpu().numpy(),
            }
        )
        data_dict["cur_res"].append(cur_res)

    # if cfg.TEST.USE_DEPTH_REFINE:
    #     process_depth_refine(data_dict, out_dict)

    poses = {}
    for res in data_dict["cur_res"]:
        pose = np.eye(4)
        pose[:3, :3] = res['R']
        pose[:3, 3] = res['t']
        poses[res['obj_id']] = pose

    return poses


if __name__ == "__main__":
    #image_paths = get_image_list(osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/test/000048/rgb"), osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/test/000048/depth"))
    yolo_predictor = YoloPredictor(
                       config_file_path=osp.join(PROJ_ROOT,"configs/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_ycbv_real_pbr_ycbv_bop_test.py"),
                       ckpt_file_path=osp.join(PROJ_ROOT,"output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_ycbv_real_pbr_ycbv_bop_test/model_final.pth"),
                       fuse=True
    )# f9772bd0df52
    gdrn_predictor = GdrnPredictor(
        config_file_path=osp.join(PROJ_ROOT,"configs/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv.py"),
        ckpt_file_path=osp.join(PROJ_ROOT,"pretrained_models/model_final_wo_optim.pth"),
        path_to_obj_models=osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/models")
    )

    obj_models = {}

    extents = {}
    idx = 0
    objs = {
            1: "002_master_chef_can",  # [1.3360, -0.5000, 3.5105]
            2: "003_cracker_box",  # [0.5575, 1.7005, 4.8050]
            3: "004_sugar_box",  # [-0.9520, 1.4670, 4.3645]
            4: "005_tomato_soup_can",  # [-0.0240, -1.5270, 8.4035]
            5: "006_mustard_bottle",  # [1.2995, 2.4870, -11.8290]
            6: "007_tuna_fish_can",  # [-0.1565, 0.1150, 4.2625]
            7: "008_pudding_box",  # [1.1645, -4.2015, 3.1190]
            8: "009_gelatin_box",  # [1.4460, -0.5915, 3.6085]
            9: "010_potted_meat_can",  # [2.4195, 0.3075, 8.0715]
            10: "011_banana",  # [-18.6730, 12.1915, -1.4635]
            11: "019_pitcher_base",  # [5.3370, 5.8855, 25.6115]
            12: "021_bleach_cleanser",  # [4.9290, -2.4800, -13.2920]
            13: "024_bowl",  # [-0.2270, 0.7950, -2.9675]
            14: "025_mug",  # [-8.4675, -0.6995, -1.6145]
            15: "035_power_drill",  # [9.0710, 20.9360, -2.1190]
            16: "036_wood_block",  # [1.4265, -2.5305, 17.1890]
            17: "037_scissors",  # [7.0535, -28.1320, 0.0420]
            18: "040_large_marker",  # [0.0460, -2.1040, 0.3500]
            19: "051_large_clamp",  # [10.5180, -1.9640, -0.4745]
            20: "052_extra_large_clamp",  # [-0.3950, -10.4130, 0.1620]
            21: "061_foam_brick",  # [-0.0805, 0.0805, -8.2435]
        }
    objs_name_to_id = {
                "002_master_chef_can": 0,  # [1.3360, -0.5000, 3.5105]
                "003_cracker_box": 1,  # [0.5575, 1.7005, 4.8050]
                "004_sugar_box": 2,  # [-0.9520, 1.4670, 4.3645]
                "005_tomato_soup_can": 3,  # [-0.0240, -1.5270, 8.4035]
                "006_mustard_bottle": 4,  # [1.2995, 2.4870, -11.8290]
                "007_tuna_fish_can": 5,  # [-0.1565, 0.1150, 4.2625]
                "008_pudding_box": 6,  # [1.1645, -4.2015, 3.1190]
                "009_gelatin_box": 7,  # [1.4460, -0.5915, 3.6085]
                "010_potted_meat_can": 8,  # [2.4195, 0.3075, 8.0715]
                "011_banana" :9 ,  # [-18.6730, 12.1915, -1.4635]
                "019_pitcher_base" :10 ,  # [5.3370, 5.8855, 25.6115]
                "021_bleach_cleanser" :11 ,  # [4.9290, -2.4800, -13.2920]
                "024_bowl" :12 ,  # [-0.2270, 0.7950, -2.9675]
                "025_mug" :13 ,  # [-8.4675, -0.6995, -1.6145]
                "035_power_drill" :14 ,  # [9.0710, 20.9360, -2.1190]
                "036_wood_block" :15 ,  # [1.4265, -2.5305, 17.1890]
                "037_scissors" :16 ,  # [7.0535, -28.1320, 0.0420]
                "040_large_marker" :17 ,  # [0.0460, -2.1040, 0.3500]
                "051_large_clamp" :18,  # [10.5180, -1.9640, -0.4745]
                "052_extra_large_clamp" :19 ,  # [-0.3950, -10.4130, 0.1620]
                "061_foam_brick" :20 ,  # [-0.0805, 0.0805, -8.2435]
            }
    obj_ids = [i for i in objs.keys()]

    models_path = {}
    textures_path = {}

    models_path_tab = []
    texture_path_tab = []
    for i, obj_name in objs.items():
        model_path = os.path.join(osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/models"), f"obj_{i:06d}.ply")
        texture_path = os.path.join(osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/models"), f"obj_{i:06d}.png")
        model = inout.load_ply(model_path, vertex_scale=0.001)
        models_path[obj_name] = model_path
        models_path_tab.append(model_path)
        textures_path[obj_name] = texture_path
        texture_path_tab.append(texture_path)
        obj_models[i] = model
        pts = model["pts"]
        xmin, xmax = np.amin(pts[:, 0]), np.amax(pts[:, 0])
        ymin, ymax = np.amin(pts[:, 1]), np.amax(pts[:, 1])
        zmin, zmax = np.amin(pts[:, 2]), np.amax(pts[:, 2])
        size_x = xmax - xmin
        size_y = ymax - ymin
        size_z = zmax - zmin
        extents[idx] = np.array([size_x, size_y, size_z], dtype="float32")
        idx += 1

    width = 640
    height = 480
    tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
    image_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
    seg_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
    ren = EGLRenderer(
        model_paths=models_path_tab,
        texture_paths=texture_path_tab,
        vertex_scale=0.001,
        use_cache=True,
        width=width,
        height=height,
    )

    with open(osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/camera_cmu.json")) as f:
        camera_json = json.load(f)
        cam = np.asarray([
            [camera_json['fx'], 0., camera_json['cx']],
            [0., camera_json['fy'], camera_json['cy']],
            [0., 0., 1.]])
        depth_scale = camera_json['depth_scale']
    extrinsic_matrix = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])
    
    vid = cv2.VideoCapture(4) #cv2.VideoCapture(0) 
    while(True): 
        start = datetime.datetime.now()
        ret, frame = vid.read() 
        if ret:
            outputs = yolo_predictor.inference(image=frame)
            inference_yolo = datetime.datetime.now()
            data_dict = preprocessing(outputs=outputs, image=frame, extents=extents, cam=cam)
            preprocess = datetime.datetime.now()

            out_dict = gdrn_predictor.inference(data_dict)
            inference_gdrn = datetime.datetime.now()

            poses = postprocessing(out_dict, objs, outputs)# gdrn_predictor.postprocessing(data_dict, out_dict)
            post_process = datetime.datetime.now()

            est_Rs = []
            est_ts = []
            est_labels = []
            for obj_name, pose in poses.items():
                est_Rs.append(pose[:3, :3])
                est_ts.append(pose[:3, 3])
                est_labels.append(objs_name_to_id[obj_name])

            im_gray = mmcv.bgr2gray(frame, keepdim=True)
            im_gray_3 = np.concatenate([im_gray, im_gray, im_gray], axis=2)

            est_poses = [np.hstack([_R, _t.reshape(3, 1)]) for _R, _t in zip(est_Rs, est_ts)]

            ren.render(
                est_labels,
                est_poses,
                K=cam,
                image_tensor=image_tensor,
                background=im_gray_3,
            )
            ren_bgr = (image_tensor[:, :, :3].detach().cpu().numpy() + 0.5).astype("uint8")

            for est_label, est_pose in zip(est_labels, est_poses):
                ren.render([est_label], [est_pose], K=cam, seg_tensor=seg_tensor)
                est_mask = (seg_tensor[:, :, 0].detach().cpu().numpy() > 0).astype("uint8")
                est_edge = get_edge(est_mask, bw=3, out_channel=1)
                ren_bgr[est_edge != 0] = np.array(mmcv.color_val("green"))    

            cv2.imshow('Main', ren_bgr)
            # cv2.imshow('YOLO res', vis_res)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
            
            vis = datetime.datetime.now()
            print(f"Inference Yolo: {(inference_yolo - start).total_seconds()*1000}ms;\n \
                Preprocess: {(preprocess - inference_yolo).total_seconds()*1000}ms;\n \
                Inference GDRN: {(inference_gdrn - preprocess).total_seconds()*1000}ms;\n \
                Post-process: {(post_process - inference_gdrn).total_seconds()*1000}ms;\n \
                Visualisation: {(vis - post_process).total_seconds()*1000}ms;\n \
                Total: {(vis - start).total_seconds()*1000}ms")