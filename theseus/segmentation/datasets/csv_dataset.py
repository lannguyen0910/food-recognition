from typing import List, Optional
import torch
import numpy as np
import pandas as pd
from PIL import Image
from .dataset import SemanticDataset
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger('main')


class CSVDataset(SemanticDataset):
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
            transform: Optional[List] = None,
            **kwargs):
        super(CSVDataset, self).__init__(**kwargs)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.csv_path = csv_path
        self.transform = transform
        self.txt_classnames = txt_classnames
        self._load_data()

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
            self.fns.append([img_name, mask_name])

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
        mask = Image.open(label_path).convert('L')
        mask = np.array(mask)  # (H,W) with each pixel value represent one class
        return mask 

    def _encode_masks(self, masks):
        """
        Input masks from _load_mask(), but in shape [B, H, W]
        Output should be one-hot encoding of segmentation masks [B, NC, H, W]
        """
        
        one_hot = torch.nn.functional.one_hot(masks.long(), num_classes=self.num_classes) # (B,H,W,NC)
        one_hot = one_hot.permute(0, 3, 1, 2) # (B,NC,H,W)
        return one_hot.float()
