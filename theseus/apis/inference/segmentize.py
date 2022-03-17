import numpy as np
import torch
import cv2
import os
import pandas as pd

from theseus.utilities.visualization.visualizer import Visualizer
from theseus.utilities.getter import (get_instance, get_instance_recursively)
from theseus.utilities.cuda import get_devices_info
from theseus.utilities.loggers import LoggerObserver, StdoutLogger
from theseus.utilities.loading import load_state_dict
from theseus.segmentation.datasets import DATALOADER_REGISTRY
from theseus.segmentation.augmentations import TRANSFORM_REGISTRY
from theseus.segmentation.models import MODEL_REGISTRY
from theseus.opt import Config

from typing import List
from PIL import Image
from datetime import datetime
from tqdm import tqdm




class SegmentationTestset(torch.utils.data.Dataset):
    """
    Custom semantic segmentation dataset on a single image path
    """
    def __init__(self, image_dir: str, txt_classnames: str, transform: List = None, **kwargs):
        self.image_dir = image_dir
        self.txt_classnames = txt_classnames
        self.transform = transform
        self.load_data()

    def load_data(self):
        df = pd.read_csv(self.txt_classnames, header=None, sep="\t")
        self.classnames = df[1].tolist()

        self.fns = []
        self.fns.append(self.image_dir)

    def __getitem__(self, index):

        image_path = self.fns[index]
        im = Image.open(image_path).convert('RGB')
        width, height = im.width, im.height

        im = np.array(im)

        if self.transform is not None:
            item = self.transform(image=im)
            im = item['image']

        return {
            "input": im,
            'img_name': os.path.basename(image_path),
            'ori_size': (width, height)
        }

    def __len__(self):
        return len(self.fns)

    def collate_fn(self, batch: List):
        imgs = torch.stack([s['input'] for s in batch])
        img_names = [s['img_name'] for s in batch]
        ori_sizes = [s['ori_size'] for s in batch]

        return {
            'inputs': imgs,
            'img_names': img_names,
            'ori_sizes': ori_sizes
        }


class SegmentationPipeline(object):
    def __init__(
        self,
        opt: Config,
        image_dir: str
    ):

        super(SegmentationPipeline, self).__init__()
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

        self.weights = opt['global']['weights']

        self.transform = get_instance_recursively(
            self.transform_cfg, registry=TRANSFORM_REGISTRY
        )

        self.dataset = SegmentationTestset(
            image_dir=image_dir,
            txt_classnames='./configs/segmentation/classes.txt',
            transform=self.transform['val'])

        CLASSNAMES = self.dataset.classnames

        self.dataloader = get_instance(
            opt['data']["dataloader"],
            registry=DATALOADER_REGISTRY,
            dataset=self.dataset,
            collate_fn=self.dataset.collate_fn
        )

        self.model = get_instance(
            self.opt["model"],
            registry=MODEL_REGISTRY,
            classnames=CLASSNAMES).to(self.device)

        if self.weights:
            state_dict = torch.load(self.weights)
            self.model = load_state_dict(self.model, state_dict, 'model')

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

        visualizer = Visualizer()
        self.model.eval()

        saved_mask_dir = os.path.join(self.savedir, 'masks')
        saved_overlay_dir = os.path.join(self.savedir, 'overlays')

        os.makedirs(saved_mask_dir, exist_ok=True)
        os.makedirs(saved_overlay_dir, exist_ok=True)

        for idx, batch in enumerate(tqdm(self.dataloader)):
            inputs = batch['inputs']
            img_names = batch['img_names']
            ori_sizes = batch['ori_sizes']

            outputs = self.model.get_prediction(batch, self.device)
            preds = outputs['masks']

            for (input, filename, ori_size) in zip(inputs, img_names, ori_sizes):
                decode_pred = visualizer.decode_segmap(preds)[:, :, ::-1]
                # decode_pred = (decode_pred * 255).astype(np.uint8)
                resized_decode_mask = cv2.resize(decode_pred, ori_size)

                # Save mask
                savepath_mask = os.path.join(saved_mask_dir, filename)
                cv2.imwrite(savepath_mask, resized_decode_mask)

                # Save overlay
                raw_image = visualizer.denormalize(input)
                ori_image = cv2.resize(raw_image, ori_size)
                overlay = ori_image * 0.7 + resized_decode_mask * 0.3
                savepath_overlay = os.path.join(saved_overlay_dir, filename)
                cv2.imwrite(savepath_overlay, overlay)

                self.logger.text(
                    f"Save image at {savepath_mask} and {savepath_overlay}", level=LoggerObserver.INFO)

        return savepath_mask