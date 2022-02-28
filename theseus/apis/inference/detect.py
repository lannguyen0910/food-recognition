import matplotlib as mpl
import cv2
import numpy as np
from theseus.segmentation.models import MODEL_REGISTRY
from theseus.segmentation.augmentations import TRANSFORM_REGISTRY
from theseus.segmentation.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from theseus.utilities.visualization.visualizer import Visualizer
import pandas as pd
import os
import torch

from theseus.utilities.getter import (get_instance, get_instance_recursively)
from theseus.utilities.cuda import get_devices_info
from theseus.utilities.loggers import LoggerObserver, StdoutLogger
from theseus.utilities.loading import load_state_dict
from theseus.classification.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY
from theseus.classification.augmentations import TRANSFORM_REGISTRY
from theseus.classification.models import MODEL_REGISTRY
from theseus.opt import Config
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from theseus.opt import Opts
from typing import List

CACHE_DIR = './.cache'

# Global model, only changes when model name changes
DETECTOR = None


mpl.use("Agg")


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

        # self.dataset = get_instance(
        #     opt['data']["dataset"],
        #     registry=DATASET_REGISTRY,
        #     transform=self.transform['val'],
        # )

        self.dataset = SegmentationTestset(
            image_dir=image_dir,
            txt_classnames='./configs/segmentation/classes.txt',
            transform=self.transform['val'])

        CLASSNAMES = self.dataset.classnames

        self.dataloader = get_instance(
            opt['data']["dataloader"],
            registry=DATALOADER_REGISTRY,
            dataset=self.dataset,
        )

        self.model = get_instance(
            self.opt["model"],
            registry=MODEL_REGISTRY,
            classnames=CLASSNAMES,
            num_classes=len(CLASSNAMES)).to(self.device)

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

        for idx, batch in enumerate(self.dataloader):
            inputs = batch['inputs']
            img_names = batch['img_names']
            ori_sizes = batch['ori_sizes']

            outputs = self.model.get_prediction(batch, self.device)
            preds = outputs['masks']

            for (input, pred, filename, ori_size) in zip(inputs, preds, img_names, ori_sizes):
                decode_pred = visualizer.decode_segmap(pred)[:, :, ::-1]
                # decode_pred = (decode_pred * 255).astype(np.uint8)
                resized_decode_mask = cv2.resize(decode_pred, ori_size)

                # Save mask
                savepath = os.path.join(saved_mask_dir, filename)
                cv2.imwrite(savepath, resized_decode_mask)

                # Save overlay
                raw_image = visualizer.denormalize(input)
                ori_image = cv2.resize(raw_image, ori_size)
                overlay = ori_image * 0.7 + resized_decode_mask * 0.3
                savepath = os.path.join(saved_overlay_dir, filename)
                cv2.imwrite(savepath, overlay)

                self.logger.text(
                    f"Save image at {savepath}", level=LoggerObserver.INFO)


if __name__ == '__main__':
    opts = Opts().parse_args()
    val_pipeline = SegmentationPipeline(opts)
    val_pipeline.inference()


class DetectionTestset():
    def __init__(self, image_dir: str, txt_classnames: str, transform: List = None, **kwargs):
        self.image_dir = image_dir  # list of cv2 images
        self.txt_classnames = txt_classnames
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
        width, height = im.width, im.height
        image_w, image_h = 640, 640

        im = np.array(im)
        ori_img = im.copy()

        if self.transform is not None:
            item = self.transform(image=im)
            im = item['image']

        return {
            "input": im,
            'img_name': os.path.basename(image_path),
            'ori_img': ori_img,
            'image_ori_w': width,
            'image_ori_h': height,
            'image_w': image_w,
            'image_h': image_h,
        }

    def __len__(self):
        return len(self.fns)

    def collate_fn(self, batch: List):
        imgs = torch.stack([s['input'] for s in batch])
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
            'img_names': img_names,
            'ori_imgs': ori_imgs,
            'image_ori_ws': image_ori_ws,
            'image_ori_hs': image_ori_hs,
            'image_ws': image_ws,
            'image_hs': image_hs,
            'img_sizes': img_sizes,
            'img_scales': img_scales
        }


def detect(args, config):
    global DETECTOR

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    test_transforms = A.Compose([
        get_resize_augmentation(
            config.image_size, keep_ratio=config.keep_ratio),
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=1.0, p=1.0),
        ToTensorV2(p=1.0)
    ])

    if args.tta:
        args.tta = TTA(
            min_conf=args.tta_conf_threshold,
            min_iou=args.tta_iou_threshold,
            postprocess_mode=args.tta_ensemble_mode)
    else:
        args.tta = None

    testset = Testset(
        config,
        args.input_path,
        transforms=test_transforms)
    testloader = DataLoader(
        testset,
        batch_size=testset.get_batch_size(),
        num_workers=2,
        pin_memory=True,
        collate_fn=testset.collate_fn
    )

    class_names, num_classes = config.names, config.nc
    class_names.insert(0, 'Background')

    if DETECTOR is None or DETECTOR.model_name != config.model_name:
        net = get_model(args, config, num_classes=num_classes)
        DETECTOR = Detector(model=net, freeze=True, device=device)

        # Print info
        print(config)

    DETECTOR.eval()

    for param in DETECTOR.parameters():
        param.requires_grad = False

    result_dict = {
        'boxes': [],
        'labels': [],
        'scores': []
    }

    empty_imgs = 0
    with tqdm(total=len(testloader)) as pbar:
        with torch.no_grad():
            for idx, batch in enumerate(testloader):
                if args.tta is not None:
                    preds = args.tta.make_tta_predictions(DETECTOR, batch)
                else:
                    preds = DETECTOR.inference_step(batch)

                for idx, outputs in enumerate(preds):
                    img_w = batch['image_ws'][idx]
                    img_h = batch['image_hs'][idx]
                    img_ori_ws = batch['image_ori_ws'][idx]
                    img_ori_hs = batch['image_ori_hs'][idx]
                    outputs = postprocessing(
                        outputs,
                        current_img_size=[img_w, img_h],
                        ori_img_size=[img_ori_ws, img_ori_hs],
                        min_iou=args.min_iou,
                        min_conf=args.min_conf,
                        max_dets=config.max_post_nms,
                        keep_ratio=config.keep_ratio,
                        output_format='xywh',
                        mode=config.fusion_mode)

                    boxes = outputs['bboxes']
                    labels = outputs['classes']
                    scores = outputs['scores']

                    for (box, label, score) in zip(boxes, labels, scores):
                        result_dict['boxes'].append(box)
                        result_dict['labels'].append(label)
                        result_dict['scores'].append(score)

                    if len(boxes) == 0:
                        empty_imgs += 1
                        boxes = None

                pbar.update(1)
                pbar.set_description(f'Empty images: {empty_imgs}')
    return result_dict
