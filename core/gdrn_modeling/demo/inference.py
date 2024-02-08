import sys
import os.path as osp
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)

import cv2
import datetime
import torch
import numpy as np

from types import SimpleNamespace
from det.yolox.engine.yolox_setup import default_yolox_setup
from det.yolox.data.data_augment import ValTransform
from core.gdrn_modeling.models.net_factory import BACKBONES
from core.gdrn_modeling.models.model_utils import (
    compute_mean_re_te,
    get_neck,
    get_geo_head,
    get_mask_prob,
    get_pnp_net,
    get_rot_mat,
    get_xyz_doublemask_region_out_dim,
)
from core.gdrn_modeling.models import GDRN_double_mask
from core.gdrn_modeling.models.GDRN_double_mask import GDRN_DoubleMask
from core.utils.my_checkpoint import load_timm_pretrained



from detectron2.config import LazyConfig
from detectron2.config.instantiate import instantiate
from mmcv import Config

import datetime 
import json
import os
from lib.pysixd import inout, misc
import copy

model = None

if __name__ == "__main__":
    vid = cv2.VideoCapture(8) 

    # YOLOx preparation
    args_yolo = SimpleNamespace(ckpt_file=osp.join(PROJ_ROOT,"pretrained_model/yolox/yolox_x.pth"),
                            config_file=osp.join(PROJ_ROOT,"configs/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_ycbv_pbr_ycbv_bop_test.py"),
                            eval_only=True,
                            fuse=False,
                            fp16=False)
    cfg_yolo = LazyConfig.load(args_yolo.config_file)
    default_yolox_setup(cfg_yolo, args_yolo)
    yolo_model = instantiate(cfg_yolo.model)
    preproc_yolo = ValTransform(legacy=False)
    yolo_model.cuda()
    yolo_model.eval()

    # GDRN preparation
    args_gdrn = SimpleNamespace(config_file=osp.join(PROJ_ROOT,"configs/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv.py"),
                                opts={'TEST.SAVE_RESULT_ONLY': True,
                                          'MODEL.WEIGHTS': osp.join(PROJ_ROOT,"pretrained_models/model_final_wo_optim.pth")},
                                TEST ={
                                    'TEST.USE_PNP': True,
                                    'TEST.USE_DEPTH_REFINE': False,
                                    'TEST.USE_COOR_Z_REFINE': False,
                                    'TEST.TEST_BBOX_TYPE': "est" # gt | est
                                },
                                eval_only=True,
                                fuse=True,
                                fp16=False,
                                resume=True,
                                vertex_scale=0.001,
                                num_gpus=1,
                                )
    
    cfg_gdrn = Config.fromfile(args_gdrn.config_file)
    cfg_gdrn.merge_from_dict(args_gdrn.opts)
    cfg_gdrn.merge_from_dict(args_gdrn.TEST)
    cfg_gdrn.SOLVER.BASE_LR = 0.0001

    objs_dir = osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/models")
    objs = {
            1: "002_master_chef_can",   
            2: "003_cracker_box",       
            3: "004_sugar_box",         
            4: "005_tomato_soup_can",   
            5: "006_mustard_bottle",    
            6: "007_tuna_fish_can",     
            7: "008_pudding_box",       
            8: "009_gelatin_box",       
            9: "010_potted_meat_can",   
            10: "011_banana",           
            11: "019_pitcher_base",     
            12: "021_bleach_cleanser",  
            13: "024_bowl",             
            14: "025_mug",              
            15: "035_power_drill",      
            16: "036_wood_block",       
            17: "037_scissors",         
            18: "040_large_marker",     
            19: "051_large_clamp",      
            20: "052_extra_large_clamp",
            21: "061_foam_brick"     
    }

    # Prepare object models
    cls_names = [i for i in objs.values()]
    obj_ids = [i for i in objs.keys()]
    obj_models = {}
    extents = {}
    idx = 0
    for i, obj_name in objs.items():
        model_path = os.path.join(objs_dir, f"obj_{i:06d}.ply")
        model = inout.load_ply(model_path, vertex_scale=args_gdrn.vertex_scale)
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

    # Prepare camera
    camera_json_path = osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/camera_cmu.json")
    with open(camera_json_path) as f:
        camera_json = json.load(f)
        cam = np.asarray([
            [camera_json['fx'], 0., camera_json['cx']],
            [0., camera_json['fy'], camera_json['cy']],
            [0., 0., 1.]])
        depth_scale = camera_json['depth_scale']

    creator_gdrn = eval(cfg_gdrn.MODEL.POSE_NET.NAME)
    net_cfg = cfg_gdrn.MODEL.POSE_NET
    backbone_cfg = net_cfg.BACKBONE

    # backbone --------------------------------
    init_backbone_args = copy.deepcopy(backbone_cfg.INIT_CFG)
    backbone_type = init_backbone_args.pop("type")
    if "timm/" in backbone_type or "tv/" in backbone_type:
        init_backbone_args["model_name"] = backbone_type.split("/")[-1]
    backbone = BACKBONES[backbone_type](**init_backbone_args)
    # Freeze the backbone
    for param in backbone.parameters():
        with torch.no_grad():
            param.requires_grad = False
    # neck --------------------------------
    neck, neck_params = get_neck(cfg_gdrn)

    # geo head -----------------------------------------------------
    geo_head, geo_head_params = get_geo_head(cfg_gdrn)

    # pnp net -----------------------------------------------
    pnp_net, pnp_net_params = get_pnp_net(cfg_gdrn)

    # build model
    gdrn_model = GDRN_DoubleMask(
        cfg_gdrn, 
        backbone, 
        neck=neck, 
        geo_head_net=geo_head, 
        pnp_net=pnp_net
    )
    load_timm_pretrained(gdrn_model.backbone, in_chans=init_backbone_args.in_chans, adapt_input_mode="custom", strict=False)
    gdrn_model.cuda()

    while(True): 
        start = datetime.datetime.now()
        ret, frame = vid.read() 
        if ret:
            start = datetime.datetime.now()
            img, _ = preproc_yolo(frame, None, cfg_yolo.test.test_size)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.cuda()
            with torch.no_grad():
                output = yolo_model(img)
                # gdrn_out = gdrn_model(img)
                print(output["det_preds"].shape)
            end = datetime.datetime.now()
            # print(f"Inference time {(end-start).total_seconds()*1000}ms")
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        #if cfg.TEST.USE_PNP:
        #    if cfg.TEST.PNP_TYPE.lower() == "ransac_pnp":
        #        return self.process_pnp_ransac(inputs, outputs, out_dict)
        #    elif cfg.TEST.PNP_TYPE.lower() == "net_iter_pnp":
        #        return self.process_net_and_pnp(inputs, outputs, out_dict, pnp_type="iter")
        #    elif cfg.TEST.PNP_TYPE.lower() == "net_ransac_pnp":Il manque une colonne pour le TD 1 j'ai l'impression de mon côté 
            
def preprocessing(outputs, image, cam):
    dataset_dict = {
        "cam": cam,
        "annotations": []
    }
    boxes = outputs[0].cpu()

    for i in range(len(boxes)):
        annot_inst = {}
        box = boxes[i].tolist()
        annot_inst["category_id"] = int(box[6])
        annot_inst["score"] = box[4] * box[5]
        annot_inst["bbox_est"] = [box[0], box[1], box[2], box[3]]
        annot_inst["bbox_mode"] = BoxMode.XYXY_ABS
        dataset_dict["annotations"].append(annot_inst)

    im_H_ori, im_W_ori = image.shape[:2]

    # other transforms (mainly geometric ones);
    # for 6d pose task, flip is not allowed in general except for some 2d keypoints methods
    im_H, im_W = image.shape[:2]  # h, w

    # NOTE: scale camera intrinsic if necessary ================================
    scale_x = im_W / im_W_ori
    scale_y = im_H / im_H_ori  # NOTE: generally scale_x should be equal to scale_y
    if "cam" in dataset_dict:
        if im_W != im_W_ori or im_H != im_H_ori:
            dataset_dict["cam"][0] *= scale_x
            dataset_dict["cam"][1] *= scale_y
        K = dataset_dict["cam"].astype("float32")
        dataset_dict["cam"] = torch.as_tensor(K)
    else:
        raise RuntimeError("cam intrinsic is missing")

    input_res = 256
    out_res = 64
    # CHW -> HWC
    coord_2d = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)
    #################################################################################
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
    start = datetime.datetime.now()
    for inst_i, inst_infos in enumerate(dataset_dict["annotations"]):
        # inherent image-level infos
        roi_infos["im_H"].append(im_H)
        roi_infos["im_W"].append(im_W)
        roi_infos["cam"].append(dataset_dict["cam"].cpu().numpy())

        # roi-level infos
        roi_infos["inst_id"].append(inst_i)
        # roi_infos["model_info"].append(inst_infos["model_info"])

        roi_cls = inst_infos["category_id"]
        roi_infos["roi_cls"].append(roi_cls)
        roi_infos["score"].append(inst_infos.get("score", 1.0))

        roi_infos["time"].append(inst_infos.get("time", 0))

        # extent
        roi_extent = self.extents[roi_cls]
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

        roi_img = self.normalize_image(self.cfg, roi_img)
        roi_infos["roi_img"].append(roi_img.astype("float32"))

        # roi_coord_2d
        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, out_res, interpolation=cv2.INTER_LINEAR
        ).transpose(
            2, 0, 1
        )  # HWC -> CHW
        roi_infos["roi_coord_2d"].append(roi_coord_2d.astype("float32"))

    end = datetime.datetime.now()
    print(f"Process image time cost: {(end - start).total_seconds()*1000}ms")

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

    for key in roi_keys:
        if key in ["roi_cls"]:
            dtype = torch.long
        else:
            dtype = torch.float32
        if key in dataset_dict:
            batch[key] = torch.cat([dataset_dict[key]], dim=0).to(device='cuda', dtype=dtype, non_blocking=True)

    batch["roi_cam"] = torch.cat([dataset_dict["cam"]], dim=0).to('cuda', non_blocking=True)
    batch["roi_center"] = torch.cat([dataset_dict["bbox_center"]], dim=0).to('cuda', non_blocking=True)
    batch["bbox_center"] = torch.cat([dataset_dict["bbox_center"]], dim=0).to('cuda', non_blocking=True)#roi_infos["bbox_center"]
    batch["cam"] = dataset_dict["cam"]

    return batch