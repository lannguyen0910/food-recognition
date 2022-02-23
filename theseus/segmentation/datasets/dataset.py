from typing import Dict, List, Optional
import os
import torch
import numpy as np
from PIL import Image

class SegmentationDataset(torch.utils.data.Dataset):
    r"""Base dataset for segmentation tasks
    """
    def __init__(self, **kwawrgs):
        self.classes_idx = {}
        self.num_classes = 0
        self.classnames = None
        self.transform = None
        self.fns = []

    def _load_data(self):
        raise NotImplementedError
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get one item
        """
        img_path, label_path = self.fns[idx]
        img = Image.open(img_path).convert('RGB')
        width, height = img.width, img.height
        img = np.array(img)
        mask = self._load_mask(label_path)
            
        if self.transform is not None:
            item = self.transform(image = img, mask = mask)
            img, mask = item['image'], item['mask']
        
        target = {}

            
        target['mask'] = mask

        return {
            "input": img, 
            'target': target,
            'img_name': os.path.basename(img_path),
            'ori_size': [width, height]
        }
    
    
    def collate_fn(self, batch):
        imgs = torch.stack([i['input'] for i in batch])
        masks = torch.stack([i['target']['mask'] for i in batch])
        
        masks = self._encode_masks(masks)
        return {
            'inputs': imgs,
            'targets': masks
        }
    
    def __len__(self) -> int:
        return len(self.fns)