
import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)

import torch
import cv2
from torch import nn
import torchvision
import datetime

from det.yolox.exp import get_exp
from det.yolox.utils import fuse_model, vis
from det.yolox.engine.yolox_setup import default_yolox_setup
from det.yolox.engine.yolox_trainer import YOLOX_DefaultTrainer
from det.yolox.data.data_augment import ValTransform

from lib.utils.time_utils import get_time_str
from core.utils.my_checkpoint import MyCheckpointer

from detectron2.config import LazyConfig
from setproctitle import setproctitle
from types import SimpleNamespace
from loguru import logger

class YoloPredictor():
    def __init__(
            self, 
            config_file_path=osp.join(PROJ_ROOT,"configs/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_ycbv_pbr_ycbv_bop_test.py"),
            ckpt_file_path=osp.join(PROJ_ROOT,"pretrained_model/yolox/yolox_x.pth"),
            fuse=True,
            fp16=False
        ):
        self.exp = get_exp(None, "yolox-x")
        self.model = self.exp.get_model()
        self.model.cuda()

        self.args = SimpleNamespace(ckpt_file=ckpt_file_path,
                                    config_file=config_file_path,
                                    eval_only=True,
                                    fuse=fuse,
                                    fp16=fp16)
        self.model = YOLOX_DefaultTrainer.build_model(self.setup())
        MyCheckpointer(self.model).resume_or_load(self.args.ckpt_file, resume=True)
        if self.args.fuse:
            logger.info("\tFusing model...")
            self.model = fuse_model(self.model)
        self.model.eval()

        self.preproc = ValTransform(legacy=False)

    def setup(self):
        """Create configs and perform basic setups."""
        cfg = LazyConfig.load(self.args.config_file)

        default_yolox_setup(cfg, self.args)
        # register_datasets_in_cfg(cfg)
        setproctitle("{}.{}".format(cfg.train.exp_name, get_time_str()))
        self.cfg = cfg.test
        return cfg

    def visual_yolo(self, output, rgb_image, class_names, cls_conf=0.35):
        # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        if output is None:
            return rgb_image
        output = output.cpu()

        bboxes = output[:, 0:4]

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(rgb_image, bboxes, scores, cls, cls_conf, class_names)
        return vis_res
            

    def postprocess(
        self, 
        det_preds, 
        num_classes, 
        conf_thre=0.7, 
        nms_thre=0.45, 
        class_agnostic=True,
        keep_single_instance=False
    ):
        box_corner = det_preds.new(det_preds.shape)
        box_corner[:, :, 0] = det_preds[:, :, 0] - det_preds[:, :, 2] / 2
        box_corner[:, :, 1] = det_preds[:, :, 1] - det_preds[:, :, 3] / 2
        box_corner[:, :, 2] = det_preds[:, :, 0] + det_preds[:, :, 2] / 2
        box_corner[:, :, 3] = det_preds[:, :, 1] + det_preds[:, :, 3] / 2
        det_preds[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(det_preds))]
        for i, image_pred in enumerate(det_preds):

            # If none are remaining => process next image
            if not image_pred.size(0):
                # logger.warn(f"image_pred.size: {image_pred.size(0)}")
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if keep_single_instance:
                instance_detections = torch.rand(num_classes, 7)
                for class_num in range(num_classes):
                    max_conf = 0
                    for detection in detections[detections[:, 6] == class_num]:
                        if detection[4] * detection[5] > max_conf:
                            instance_detections[class_num] = detection
                            max_conf = detection[4] * detection[5]
                detections = instance_detections
            if not detections.size(0):
                # logger.warn(f"detections.size(0) {detections.size(0)} num_classes: {num_classes} conf_thr: {conf_thre} nms_thr: {nms_thre}")
                continue

            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    nms_thre,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    nms_thre,
                )

            detections = detections[nms_out_index]
            detections = torch.tensor(detections[detections[:, 6].argsort()])
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))
        return output

    def inference(self, image):
        """
        Preprocess input image, run inference and postprocess the output.
        Args:
            image: rgb image
        Returns:
            postprocessed output
        """
        img, _ = self.preproc(image, None, self.cfg.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.cuda()

        with torch.no_grad():
            outputs = self.model(img, cfg=self.cfg)
            outputs = self.postprocess(outputs["det_preds"],
                                  self.cfg.num_classes,
                                  class_agnostic=True,
                                  keep_single_instance=True)

        return outputs

if __name__ == "__main__":
    predictor = YoloPredictor(
        config_file_path=osp.join(PROJ_ROOT,"configs/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_ycbv_real_pbr_ycbv_bop_test.py"),
        ckpt_file_path=osp.join(PROJ_ROOT,"/home/achapin/Documents/gdrnpp_bop2022/output/yolox/bop_pbr/yolox_x_640_augCozyAAEhsv_ranger_30_epochs_ycbv_real_pbr_ycbv_bop_test/model_final.pth"),
        fuse=True
    )
    img_path = osp.join(PROJ_ROOT,"datasets/BOP_DATASETS/ycbv/test/000048/rgb/000001.png")
    img = cv2.imread(img_path)
    result = predictor.inference(img)
    predictor.visual_yolo(result[0], img, ["002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box", 
    "005_tomato_soup_can",  
    "006_mustard_bottle",  
    "007_tuna_fish_can",  
    "008_pudding_box",  
    "009_gelatin_box",  
    "010_potted_meat_can",  
    "011_banana",  
    "019_pitcher_base",  
    "021_bleach_cleanser",  
    "024_bowl", 
    "025_mug",
    "035_power_drill", 
    "036_wood_block", 
    "037_scissors", 
    "040_large_marker",  
    "051_large_clamp",  
    "052_extra_large_clamp", 
    "061_foam_brick"])
    