from typing import Dict, List, Optional
import os
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from .dataset import SegmentationDataset
from theseus.segmentation.augmentations.mosaic import Mosaic
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger('main')


class CSVDatasetWithMosaic(SegmentationDataset):
    r"""CSVDataset multi-labels segmentation dataset

    Reads in .csv file with structure below:
        filename   | label
        ---------- | -----------
        <img1>.jpg | <mask1>.jpg

    image_dir: `str`
        path to directory contains images
    mask_dir: `str`
        path to directory contains masks
    transform: Optional[List]
        transformatin functions
        
    """
    def __init__(
            self, 
            image_dir: str, 
            mask_dir: str, 
            csv_path: str, 
            txt_classnames: str,
            mosaic_size: int, 
            mosaic_prob: float = 0.3,
            transform: Optional[List] = None,
            **kwargs):
        super(CSVDatasetWithMosaic, self).__init__(**kwargs)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.csv_path = csv_path
        self.transform = transform
        self.txt_classnames = txt_classnames
        self.mosaic = Mosaic(mosaic_size, mosaic_size)
        self.mosaic_size = mosaic_size
        self.mosaic_prob = mosaic_prob
        self._load_data()

    def load_mosaic(self, index:int):
        indexes = [index] + [random.randint(0, len(self.fns) - 1) for _ in range(3)]
        images_list = []
        masks_list = []

        for index in indexes:
            img_path, label_path = self.fns[index]
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            mask = self._load_mask(label_path)
            images_list.append(img)
            masks_list.append(mask)

        result_image, result_mask = self.mosaic(
            images_list, 
            masks_list)
            
        return result_image, result_mask

    def __getitem__(self, idx: int) -> Dict:
        """
        Get one item
        """

        if random.uniform(0,1) <= self.mosaic_prob:
            img, mask = self.load_mosaic(idx)
            width, height = self.mosaic_size, self.mosaic_size
            basename = None
        else:
            img_path, label_path = self.fns[idx]
            img = Image.open(img_path).convert('RGB')
            width, height = img.width, img.height
            img = np.array(img)
            mask = self._load_mask(label_path)

            basename = os.path.basename(img_path)
            
        if self.transform is not None:
            item = self.transform(image = img, mask = mask)
            img, mask = item['image'], item['mask']
        
        target = {}
            
        target['mask'] = mask

        return {
            "input": img, 
            'target': target,
            'img_name': basename,
            'ori_size': [width, height]
        }

    def _load_data(self):
        """
        Read data from csv and load into memory
        """

        with open(self.txt_classnames, 'r') as f:
            self.classnames = f.read().splitlines()
        
        # Mapping between classnames and indices
        for idx, classname in enumerate(self.classnames):
            self.classes_idx[classname] = idx
        self.num_classes = len(self.classnames)
        
        df = pd.read_csv(self.csv_path)
        for idx, row in df.iterrows():
            img_name, mask_name = row
            image_path = os.path.join(self.image_dir,img_name)
            mask_path = os.path.join(self.mask_dir, mask_name)
            self.fns.append([image_path, mask_path])

    def _calculate_classes_dist(self):
        LOGGER.text("Calculating class distribution...", LoggerObserver.DEBUG)
        self.classes_dist = []
        for _, mask_path in self.fns:
            mask = self._load_mask(mask_path)
            unique_ids = np.unique(mask).tolist()

            # A hack, because classes distribute fewer for higher index
            label = max(unique_ids)
            self.classes_dist.append(label)
        return self.classes_dist


    def _load_mask(self, label_path):
        mask = Image.open(label_path).convert('RGB')
        mask = np.array(mask)[:,:,::-1] # (H,W,3)
        mask = np.argmax(mask, axis=-1)  # (H,W) with each pixel value represent one class

        return mask 

    def _encode_masks(self, masks):
        """
        Input masks from _load_mask(), but in shape [B, H, W]
        Output should be one-hot encoding of segmentation masks [B, NC, H, W]
        """
        
        one_hot = torch.nn.functional.one_hot(masks.long(), num_classes=self.num_classes) # (B,H,W,NC)
        one_hot = one_hot.permute(0, 3, 1, 2) # (B,NC,H,W)
        return one_hot.float()