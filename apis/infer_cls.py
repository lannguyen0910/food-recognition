from typing import List, Optional, Tuple

import matplotlib as mpl
mpl.use("Agg")
from theseus.opt import Opts

import os
from datetime import datetime
from tqdm import tqdm
import torch
from theseus.opt import Config
from theseus.classification.models import MODEL_REGISTRY
from theseus.classification.augmentations import TRANSFORM_REGISTRY
from theseus.classification.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from theseus.utilities.loading import load_state_dict
from theseus.utilities.loggers import LoggerObserver, StdoutLogger
from theseus.utilities.cuda import get_devices_info
from theseus.utilities.getter import (get_instance, get_instance_recursively)

import os
import pandas as pd


class TestPipeline(object):
    def __init__(
            self,
            opt: Config
        ):

        super(TestPipeline, self).__init__()
        self.opt = opt

        self.debug = opt['global']['debug']
        self.logger = LoggerObserver.getLogger("main") 
        self.savedir = os.path.join(opt['global']['save_dir'], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
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

        self.dataset = get_instance(
            opt['data']["dataset"],
            registry=DATASET_REGISTRY,
            transform=self.transform['val'],
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
        self.logger.text(f"Number of test sample: {len(self.dataset)}", level=LoggerObserver.INFO)
        self.logger.text(f"Everything will be saved to {self.savedir}", level=LoggerObserver.INFO)

    def inference(self):
        self.infocheck()
        self.logger.text("Inferencing...", level=LoggerObserver.INFO)

        df_dict = {
            'filename': [],
            'label': [],
            'score': []
        }
        
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
        

if __name__ == '__main__':
    opts = Opts().parse_args()
    val_pipeline = TestPipeline(opts)
    val_pipeline.inference()

        
