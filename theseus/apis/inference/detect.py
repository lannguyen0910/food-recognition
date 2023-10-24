import os
import torch

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

from tqdm import tqdm
from datetime import datetime
from typing import List, Any

CACHE_DIR = './weights'



class DetectionTestset():
    """
    Custom detection dataset on a single image path
    """

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
        self.model_name = opt["model"]["name"].lower()
        self.device_name = opt['global']['device']
        self.device = torch.device(self.device_name)

        # Dection arguments defined in modules.py
        self.args = input_args

        # Whether to use test-time augmentation
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
            transform=self.transform['val'],
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
        os.makedirs(self.savedir, exist_ok=True)

        boxes_result = []
        labels_result = []
        scores_result = []

        if self.model_name.startswith("yolov8"):
            image = self.dataset.image_dir
            preds = self.model.get_prediction(image)

            for _, outputs in enumerate(preds):
                boxes = outputs['bboxes']
                labels = outputs['classes']
                scores = outputs['scores']

                if len(boxes) == 0:
                    continue

                boxes_result.append(boxes)
                labels_result.append(labels)
                scores_result.append(scores)

        else:
            for _, batch in enumerate(tqdm(self.dataloader)):
                if self.tta is not None:
                    preds = self.tta.make_tta_predictions(
                        self.model, batch, self.device)
                else:    
                    preds = self.model.get_prediction(batch, self.device)

                for _, outputs in enumerate(preds):
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
