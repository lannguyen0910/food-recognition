from models import *
from utilities.trainer import *
from utilities.configs import *
from utilities.augmentations import *
import os
import cv2
import math
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR, ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts

from utilities.utils.utils import draw_boxes_v2
from utilities.utils.cuda import NativeScaler, get_devices_info
from utilities.utils.postprocess import change_box_order, postprocessing

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .random_seed import seed_everything


def get_instance(config, **kwargs):
    # Inherited from https://github.com/vltanh/pytorch-template
    assert 'name' in config
    config.setdefault('args', {})
    if config['args'] is None:
        config['args'] = {}
    return globals()[config['name']](**config['args'], **kwargs)


def get_lr_policy(opt_config):
    optimizer_params = {}
    lr = opt_config['lr'] if 'lr' in opt_config.keys() else None
    if opt_config["name"] == 'sgd':
        optimizer = SGD
        optimizer_params = {
            'lr': lr,
            'weight_decay': opt_config['weight_decay'],
            'momentum': opt_config['momentum'],
            'nesterov': True}
    elif opt_config["name"] == 'adam':
        optimizer = AdamW
        optimizer_params = {
            'lr': lr,
            'weight_decay': opt_config['weight_decay'],
            'betas': (opt_config['momentum'], 0.999)}
    return optimizer, optimizer_params


def get_lr_scheduler(optimizer, lr_config, **kwargs):

    scheduler_name = lr_config["name"]
    step_per_epoch = False

    if scheduler_name == '1cycle-yolo':
        def one_cycle(y1=0.0, y2=1.0, steps=100):
            # lambda function for sinusoidal ramp from y1 to y2
            return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

        lf = one_cycle(1, 0.158, kwargs['num_epochs'])  # cosine 1->hyp['lrf']
        scheduler = LambdaLR(optimizer, lr_lambda=lf)
        step_per_epoch = True

    elif scheduler_name == '1cycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.001,
            epochs=kwargs['num_epochs'],
            steps_per_epoch=int(
                len(kwargs["trainset"]) / kwargs["batch_size"]),
            pct_start=0.1,
            anneal_strategy='cos',
            final_div_factor=10**5)
        step_per_epoch = False

    elif scheduler_name == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=1,
            verbose=False,
            threshold=0.0001,
            threshold_mode='abs',
            cooldown=0,
            min_lr=1e-8,
            eps=1e-08
        )
        step_per_epoch = True

    elif scheduler_name == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs['num_epochs'],
            T_mult=1,
            eta_min=0.0001,
            last_epoch=-1,
            verbose=False
        )
        step_per_epoch = False
    return scheduler, step_per_epoch


def get_dataset_and_dataloader(config):

    if config.model_name.startswith('efficientdet'):
        box_format = 'yxyx'  # Output of __getitem__ method

        def collate_fn(self, batch):
            imgs = torch.stack([s['img'] for s in batch])
            targets = [s['target'] for s in batch]
            img_ids = [s['img_id'] for s in batch]
            img_names = [s['img_name'] for s in batch]
            img_scales = torch.tensor([1.0]*len(batch), dtype=torch.float)
            img_sizes = torch.tensor(
                [imgs[0].shape[-2:]]*len(batch), dtype=torch.float)
            ori_sizes = [s['ori_size'] for s in batch]

            return {
                'imgs': imgs,
                'targets': targets,
                'img_ids': img_ids,
                'img_names': img_names,
                'img_sizes': img_sizes,
                'img_scales': img_scales,
                'ori_sizes': ori_sizes}

    elif config.model_name.startswith('yolo'):
        box_format = 'xyxy'  # Output of __getitem__ method

        def collate_fn(self, batch):
            imgs = torch.stack([s['img'] for s in batch], dim=0)
            targets = [s['target'] for s in batch]  # box in center xywh format
            img_names = [s['img_name'] for s in batch]
            img_ids = [s['img_id'] for s in batch]
            ori_sizes = [s['ori_size'] for s in batch]
            img_scales = torch.tensor([1.0]*len(batch), dtype=torch.float)
            img_sizes = torch.tensor(
                [imgs[0].shape[-2:]]*len(batch), dtype=torch.float)
            img_size = imgs[0].shape[-1]
            targets_out = []
            for idx, item in enumerate(targets):
                # Convert to center xywh
                cxcy_boxes = change_box_order(item['boxes'], order='xyxy2cxcy')
                cxcy_boxes = cxcy_boxes / img_size  # normalize
                num_boxes = cxcy_boxes.shape[0]
                labels_out = torch.zeros([num_boxes, 6])
                labels = item['labels'].unsqueeze(1)

                # Yolo need class starts at 0
                labels = labels - 1
                out_anns = torch.cat([labels, cxcy_boxes], dim=1)
                labels_out[:, 1:] = out_anns[:, :]
                labels_out[:, 0] = idx
                targets_out.append(labels_out)
            targets_out = torch.cat(targets_out, dim=0)

            return {
                'imgs': imgs,
                'targets': targets,
                'yolo_targets': targets_out,
                'img_ids': img_ids,
                'img_names': img_names,
                'img_scales': img_scales,
                'img_sizes': img_sizes,
                'ori_sizes': ori_sizes
            }

    CocoDataset.collate_fn = collate_fn

    trainset = CocoDataset(
        config=config,
        root_dir=os.path.join('data', config.project_name, config.train_imgs),
        ann_path=os.path.join('data', config.project_name, config.train_anns),
        train=True)

    valset = CocoDataset(
        config=config,
        root_dir=os.path.join('data', config.project_name, config.val_imgs),
        ann_path=os.path.join('data', config.project_name, config.val_anns),
        train=False)

    trainset.set_box_format(box_format)
    valset.set_box_format(box_format)
    config.box_format = box_format

    trainloader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=trainset.collate_fn,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=True)

    valloader = DataLoader(
        valset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=valset.collate_fn,
        num_workers=config.num_workers,
        pin_memory=True)

    return trainset, valset, trainloader, valloader
