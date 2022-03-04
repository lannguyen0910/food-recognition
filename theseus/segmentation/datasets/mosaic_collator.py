import random
from typing import Dict, List, Tuple
import torch
import numpy as np
from theseus.base.datasets.collator import BaseCollator

class MosaicCollator(BaseCollator):
    """
    Mosaic augmentation for image segmentation

    width: `int`
        output width
    height: `int`
        output height
    scale_range: `Tuple[float, float]` 
        scale for each tile of mosaic

    """
    def __init__(self, scale_range: Tuple[float, float] = (0.3, 0.7), p=0.5, **kwargs) -> None:
        self.scale_range = scale_range
        self.p = p

    def __call__(self, batch: List[Dict]):
        """ 
        batch: `List[Dict]`
            batch of tensor images and mask # (B,3,H,W), (B,NC,H,W)
        return: ` List[Dict]`
            new batch of image and mask
        """

        if random.random() > self.p:
            return batch

        set_images = batch['inputs']
        set_masks = batch['targets']
        batch_size, channels, height, width = set_images.shape
        num_classes = set_masks.shape[1]
        
        result_images = []
        result_masks = []
        for b in range(batch_size):
            current_image = set_images[b]
            current_mask = set_masks[b]
            result_image = torch.zeros((channels, height, width))
            result_mask = torch.zeros((num_classes, height, width))
            
            scale_x = self.scale_range[0] + random.random() * (self.scale_range[1] - self.scale_range[0])
            scale_y = self.scale_range[0] + random.random() * (self.scale_range[1] - self.scale_range[0])
            divid_point_x = int(scale_x * width)
            divid_point_y = int(scale_y * height)

            candidate_indices = np.random.choice(np.setdiff1d(range(batch_size), b), 3, replace=False)
            candidate_images = set_images[candidate_indices, :]
            candidate_masks = set_masks[candidate_indices, :]

            candidate_images = torch.cat([candidate_images, current_image.unsqueeze(0)], dim=0)
            candidate_masks = torch.cat([candidate_masks, current_mask.unsqueeze(0)], dim=0)
  
            shuffling_id = torch.randperm(candidate_images.size()[0])
            candidate_images = candidate_images[shuffling_id]
            candidate_masks = candidate_masks[shuffling_id]

            for i, (img, mask) in enumerate(zip(candidate_images, candidate_masks)):
                if i == 0:  # top-left
                    result_image[:, :divid_point_y, :divid_point_x] = img[:, :divid_point_y, :divid_point_x]
                    result_mask[:, :divid_point_y, :divid_point_x] = mask[:, :divid_point_y, :divid_point_x]

                elif i == 1:  # top-right
                    result_image[:, :divid_point_y, divid_point_x:width] = img[:, :divid_point_y, divid_point_x:width]
                    result_mask[:, :divid_point_y, divid_point_x:width] = mask[:, :divid_point_y, divid_point_x:width]
                    
                elif i == 2:  # bottom-left
                    result_image[:, divid_point_y:height, :divid_point_x] = img[:, divid_point_y:height, :divid_point_x]
                    result_mask[:, divid_point_y:height, :divid_point_x] = mask[:, divid_point_y:height, :divid_point_x]
                
                else:  # bottom-right
                    result_image[:, divid_point_y:height, divid_point_x:width] = img[:, divid_point_y:height, divid_point_x:width]
                    result_mask[:, divid_point_y:height, divid_point_x:width] = mask[:, divid_point_y:height, divid_point_x:width]

            result_images.append(result_image)
            result_masks.append(result_mask)
            
        batch['inputs'] = torch.stack(result_images, dim=0)
        batch['targets'] = torch.stack(result_masks, dim=0)

        return batch