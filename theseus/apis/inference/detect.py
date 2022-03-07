import matplotlib as mpl
import numpy as np
import os
import torch
from albumentations.pytorch.transforms import ToTensorV2

from theseus.detection.models import MODEL_REGISTRY
from theseus.detection.augmentations import *
from theseus.base.datasets import DATALOADER_REGISTRY
from theseus.utilities.getter import (get_instance, get_instance_recursively)
from theseus.utilities.cuda import get_devices_info
from theseus.utilities.loggers import LoggerObserver, StdoutLogger
from theseus.utilities.loading import load_state_dict
from theseus.detection.augmentations import TRANSFORM_REGISTRY, TTA
from theseus.detection.models import MODEL_REGISTRY
from theseus.opt import Config
from theseus.utilities import postprocessing

from tqdm import tqdm
from datetime import datetime
from PIL import Image
from typing import List, Any

CACHE_DIR = './weights'

mpl.use("Agg")


class DetectionTestset():
    def __init__(self, image_dir: str, transform: List = None, **kwargs):
        self.image_dir = image_dir  # list of cv2 images
        self.transform = transform
        self.fns = []
        self.load_data()

    def load_data(self):
        """
        Load filepaths into memory
        """
        if os.path.isdir(self.image_dir):  # path to image folder
            paths = sorted(os.listdir(self.image_dir))
            for path in paths:
                self.fns.append(
                    os.path.join(self.image_dir, path))
        elif os.path.isfile(self.image_dir):  # path to single image
            self.fns.append(self.image_dir)

    def __getitem__(self, index: int):
        """
        Get an item from memory
        """
        image_path = self.fns[index]

        im = cv2.imread(image_path)[..., ::-1]
        image_w, image_h = 640, 640
        ori_height, ori_width, c = im.shape
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        ori_img = im.copy()
        clone_img = im.copy()

        if self.transform is not None:
            item = self.transform(image=clone_img)
            clone_img = item['image']

        return {
            "input": im,
            "torch_input": clone_img,
            'img_name': image_path,
            'ori_img': ori_img,
            'image_ori_w': ori_width,
            'image_ori_h': ori_height,
            'image_w': image_w,
            'image_h': image_h,
        }

    def __len__(self):
        return len(self.fns)

    def collate_fn(self, batch: List):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None

        imgs = [s['input'] for s in batch]
        torch_imgs = torch.stack([s['torch_input'] for s in batch])
        img_names = [s['img_name'] for s in batch]
        ori_imgs = [s['ori_img'] for s in batch]
        image_ori_ws = [s['image_ori_w'] for s in batch]
        image_ori_hs = [s['image_ori_h'] for s in batch]
        image_ws = [s['image_w'] for s in batch]
        image_hs = [s['image_h'] for s in batch]
        img_scales = torch.tensor([1.0]*len(batch), dtype=torch.float)
        img_sizes = torch.tensor(
            [imgs[0].shape[-2:]]*len(batch), dtype=torch.float)

        return {
            'inputs': imgs,
            'torch_inputs': torch_imgs,
            'img_names': img_names,
            'ori_imgs': ori_imgs,
            'image_ori_ws': image_ori_ws,
            'image_ori_hs': image_ori_hs,
            'image_ws': image_ws,
            'image_hs': image_hs,
            'img_sizes': img_sizes,
            'img_scales': img_scales
        }


class DetectionPipeline(object):
    def __init__(
        self,
        opt: Config,
        input_args: Any  # Input arguments from users
    ):

        super(DetectionPipeline, self).__init__()
        self.opt = opt

        self.debug = opt['global']['debug']
        self.logger = LoggerObserver.getLogger("main")
        self.savedir = os.path.join(
            opt['global']['save_dir'], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(self.savedir, exist_ok=True)

        stdout_logger = StdoutLogger(__name__, self.savedir, debug=self.debug)
        self.logger.subscribe(stdout_logger)
        self.logger.text(self.opt, level=LoggerObserver.INFO)

        self.transform_cfg = Config.load_yaml(opt['global']['cfg_transform'])
        self.device_name = opt['global']['device']
        self.device = torch.device(self.device_name)

        self.args = input_args

        if input_args.tta:
            self.tta = TTA(
                min_conf=input_args.tta_conf_threshold,
                min_iou=input_args.tta_iou_threshold,
                postprocess_mode=input_args.tta_ensemble_mode)
        else:
            self.tta = None

        self.transform = get_instance_recursively(
            self.transform_cfg, registry=TRANSFORM_REGISTRY
        )

        # self.transform = A.Compose([
        #     get_resize_augmentation(
        #         image_size=[640, 640], keep_ratio=True),
        #     A.Normalize(mean=[0.0, 0.0, 0.0], std=[
        #                 1.0, 1.0, 1.0], max_pixel_value=1.0, p=1.0),
        #     ToTensorV2(p=1.0)
        # ])

        self.dataset = DetectionTestset(
            image_dir=input_args.input_path,
            transform=self.transform['val'],
            # transform=self.transform
        )

        self.class_names = opt['global']['class_names']

        self.dataloader = get_instance(
            opt['data']["dataloader"],
            registry=DATALOADER_REGISTRY,
            dataset=self.dataset,
            collate_fn=self.dataset.collate_fn
        )

        self.model = get_instance(
            opt['model'],
            registry=MODEL_REGISTRY,
            weight=input_args.weight,
            min_iou=input_args.min_iou,
            min_conf=input_args.min_conf,
        ).to(self.device)

        if input_args.weight:
            state_dict = torch.load(input_args.weight)
            self.model = load_state_dict(self.model, state_dict,
                                         'model', is_detection=True)

    def infocheck(self):
        device_info = get_devices_info(self.device_name)
        self.logger.text("Using " + device_info, level=LoggerObserver.INFO)
        self.logger.text(
            f"Number of test sample: {len(self.dataset)}", level=LoggerObserver.INFO)
        self.logger.text(
            f"Everything will be saved to {self.savedir}", level=LoggerObserver.INFO)

    @torch.no_grad()
    def inference(self):
        self.infocheck()
        self.logger.text("Inferencing...", level=LoggerObserver.INFO)

        self.model.eval()

        boxes_result = []
        labels_result = []
        scores_result = []

        for idx, batch in enumerate(tqdm(self.dataloader)):

            os.makedirs(self.savedir, exist_ok=True)

            if self.tta is not None:
                preds = self.tta.make_tta_predictions(
                    self.model, batch, self.device, self.args.weight)
            else:
                preds = self.model.get_prediction(batch, self.device)

            for idx, outputs in enumerate(preds):
                # img_w = batch['image_ws'][idx]
                # img_h = batch['image_hs'][idx]
                # img_ori_ws = batch['image_ori_ws'][idx]
                # img_ori_hs = batch['image_ori_hs'][idx]

                # outputs = postprocessing(
                #     outputs,
                #     current_img_size=[img_w, img_h],
                #     ori_img_size=[img_ori_ws, img_ori_hs],
                #     min_iou=self.args.min_iou,
                #     min_conf=self.args.min_conf,
                #     keep_ratio=True,
                #     output_format='xywh',
                #     mode='nms'
                # )

                boxes = outputs['bboxes']
                labels = outputs['classes']
                scores = outputs['scores']

                if len(boxes) == 0:
                    continue

                boxes_result.append(boxes)
                labels_result.append(labels)
                scores_result.append(scores)

        return {
            "boxes": boxes_result,
            "labels": labels_result,
            "scores": scores_result}


""" 
Result after inference:  <models.common.Detections object at 0x7ff7e759c2d0>

TTA
[00:00<?, ?it/s]Result after inference:  tensor([[[7.58108e+00, 3.29767e+00, 8.19291e+00,  ..., 1.63066e-03, 1.83365e-03, 3.13231e-03],
         [1.48919e+01, 2.81062e+00, 9.60818e+00,  ..., 1.08319e-03, 8.73214e-04, 1.84057e-03],
         [1.97235e+01, 3.86513e+00, 1.48165e+01,  ..., 7.01686e-04, 6.10586e-04, 8.99242e-04],
         ...,
         [5.62207e+02, 5.96428e+02, 1.67612e+02,  ..., 6.05156e-02, 4.12644e-02, 1.58415e-02],
         [5.92910e+02, 5.98386e+02, 1.88376e+02,  ..., 1.05861e-01, 6.31578e-02, 2.09839e-02],
         [6.30867e+02, 6.02710e+02, 1.75896e+02,  ..., 1.14207e-01, 6.17458e-02, 1.67981e-02]]], device='cuda:0')
Result after inference tta:  [[[     7.5811      3.2977      8.1929 ...   0.0016307   0.0018337   0.0031323]
  [     14.892      2.8106      9.6082 ...   0.0010832  0.00087321   0.0018406]
  [     19.723      3.8651      14.816 ...  0.00070169  0.00061059  0.00089924]
  ...
  [     562.21      596.43      167.61 ...    0.060516    0.041264    0.015841]
  [     592.91      598.39      188.38 ...     0.10586    0.063158    0.020984]
  [     630.87      602.71       175.9 ...     0.11421    0.061746    0.016798]]]
"""
