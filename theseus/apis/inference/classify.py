import pandas as pd
import os
import torch

from theseus.utilities.getter import (get_instance, get_instance_recursively)
from theseus.utilities.cuda import get_devices_info
from theseus.utilities.loggers import LoggerObserver, StdoutLogger
from theseus.utilities.loading import load_state_dict
from theseus.classification.datasets import DATALOADER_REGISTRY
from theseus.classification.augmentations import TRANSFORM_REGISTRY
from theseus.classification.models import MODEL_REGISTRY
from theseus.opt import Config
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from typing import List


class ClassificationTestset():
    """
    Custom classification dataset on a list of cropped images
    """

    def __init__(self, image_dir: list, txt_classnames: str, transform: List = None, **kwargs):
        self.img_list = image_dir  # list of cv2 images
        self.txt_classnames = txt_classnames
        self.transform = transform
        self.load_data()

    def load_data(self):
        """
        Load filepaths into memory
        """
        with open(self.txt_classnames, 'r') as f:
            self.classnames = f.read().splitlines()

    def __getitem__(self, index: int):
        """
        Get an item from memory
        """
        image_path = self.img_list[index]
        im = Image.open(image_path).convert('RGB')
        width, height = im.width, im.height

        if self.transform is not None:
            im = self.transform(im)

        return {
            "input": im,
            'img_name': os.path.basename(image_path),
            'ori_size': [width, height]
        }

    def __len__(self):
        return len(self.img_list)

    def collate_fn(self, batch: List):
        imgs = torch.stack([s['input'] for s in batch])
        img_names = [s['img_name'] for s in batch]

        return {
            'inputs': imgs,
            'img_names': img_names
        }


class ClassificationPipeline(object):
    def __init__(
        self,
        opt: Config,
        image_dir: list
    ):

        super(ClassificationPipeline, self).__init__()
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

        self.dataset = ClassificationTestset(
            image_dir=image_dir,
            txt_classnames='./configs/classification/classes.txt',
            transform=self.transform['val']
        )

        CLASSNAMES = self.dataset.classnames

        self.dataloader = get_instance(
            opt['data']["dataloader"],
            registry=DATALOADER_REGISTRY,
            dataset=self.dataset,
            collate_fn=self.dataset.collate_fn
        )

        self.model = get_instance(
            opt["model"],
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

        df_dict = {
            'filename': [],
            'label': [],
            'score': []
        }

        self.model.eval()

        for idx, batch in enumerate(tqdm(self.dataloader)):
            img_names = batch['img_names']
            outputs = self.model.get_prediction(batch, self.device)
            preds = outputs['names']
            probs = outputs['confidences']

            for (filename, pred, prob) in zip(img_names, preds, probs):
                df_dict['filename'].append(filename)
                df_dict['label'].append(pred)
                df_dict['score'].append(prob)

        df = pd.DataFrame(df_dict)
        savepath = os.path.join(self.savedir, 'prediction.csv')
        df.to_csv(savepath, index=False)

        return df_dict
