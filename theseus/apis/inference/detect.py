import matplotlib as mpl
import cv2
import numpy as np
from theseus.segmentation.models import MODEL_REGISTRY
from theseus.segmentation.augmentations import TRANSFORM_REGISTRY
from theseus.segmentation.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
import pandas as pd
import os
import torch

from theseus.utilities.getter import (get_instance, get_instance_recursively)
from theseus.utilities.cuda import get_devices_info
from theseus.utilities.loggers import LoggerObserver, StdoutLogger
from theseus.utilities.loading import load_state_dict
from theseus.detection.augmentations import TRANSFORM_REGISTRY, TTA
from theseus.detection.models import MODEL_REGISTRY
from theseus.opt import Config
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from theseus.opt import Opts
from typing import List, Any

CACHE_DIR = './weights'

# Global model, only changes when model name changes
DETECTOR = None

mpl.use("Agg")


class DetectionTestset():
    def __init__(self, image_dir: str, transform: List = None, **kwargs):
        self.image_dir = image_dir  # list of cv2 images
        self.transform = transform
        self.load_data()

    def load_data(self):
        """
        Load filepaths into memory
        """
        self.fns = []
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
        im = Image.open(image_path).convert('RGB')
        # width, height = im.width, im.height

        im = np.array(im)
        ori_img = im.copy()

        if self.transform is not None:
            item = self.transform(image=im)
            im = item['image']

        return {
            "input": im,
            'img_name': os.path.basename(image_path),
            'ori_img': ori_img,
        }

    def __len__(self):
        return len(self.fns)

    def collate_fn(self, batch: List):
        imgs = torch.stack([s['input'] for s in batch])
        img_names = [s['img_name'] for s in batch]
        ori_imgs = [s['ori_img'] for s in batch]

        return {
            'inputs': imgs,
            'img_names': img_names,
            'ori_imgs': ori_imgs,
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

        self.weights = input_args.weight

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

        self.dataset = DetectionTestset(
            image_dir=input_args.input_path,
            transform=self.transform['val']
        )

        self.class_names = opt['global']['class_names']

        self.dataloader = get_instance(
            opt['data']["dataloader"],
            registry=DATALOADER_REGISTRY,
            dataset=self.dataset,
            collate_fn=self.dataset.collate_fn
        )

        self.model = get_instance(
            input_args.model_name,
            registry=MODEL_REGISTRY,
            min_iou=input_args.min_iou,
            min_conf=input_args.min_conf,
            max_det=self.opt['global']['max_det']
        ).to(self.device)

        global DETECTOR
        # Not to load the same detection model again
        if DETECTOR is None or DETECTOR.name != self.model.name:
            DETECTOR = self.model

            if self.weights:
                state_dict = torch.load(self.weights)
                DETECTOR = load_state_dict(self.model, state_dict, 'model')

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

        DETECTOR.eval()

        boxes_result = []
        labels_result = []
        scores_result = []

        for idx, batch in enumerate(tqdm(self.dataloader)):

            os.makedirs(self.savedir, exist_ok=True)

            if self.tta is not None:
                preds = self.tta.make_tta_predictions(DETECTOR, batch)
            else:
                preds = DETECTOR.get_prediction(batch, self.device)

            # for (bboxes, filename, labels, scores) in zip(outputs['bboxes'], img_names, outputs['labels'], outputs['scores']):
            #     savepath = os.path.join(self.savedir, filename)
            #     # Write bboxes and texts to image and save to "savepath"
            #     visualizer.draw_bbox(savepath, bboxes, labels, scores)

            #     self.logger.text(
            #         f"Save image at {savepath}", level=LoggerObserver.INFO)
            for id, outputs in enumerate(preds):
                boxes = outputs['bboxes']

                # Here, labels start from 1, subtract 1
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
