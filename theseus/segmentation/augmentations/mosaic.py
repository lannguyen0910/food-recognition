import random
from typing import List, Tuple
import numpy as np
from albumentations import RandomCrop, Compose, Resize 

class Mosaic:
    """
    Mosaic augmentation for image segmentation

    width: `int`
        output width
    height: `int`
        output height
    scale_range: `Tuple[float, float]` 
        scale for each tile of mosaic

    """
    def __init__(self, width: int, height: int, scale_range: Tuple[float, float] = (0.3, 0.7)) -> None:
        self.width = width
        self.height = height
        self.scale_range = scale_range

    def get_resize(self, image: np.array, mask: np.array, width: int, height: int):
        """
        Random resize and crop image and mask
        """
        max_width = max(width, image.shape[1])
        max_height = max(height, image.shape[0])

        transforms = Compose([
            Resize(max_height, max_width),
            RandomCrop(height, width)
        ])

        item = transforms(image = image, mask = mask)
        return item['image'], item['mask']

    def __call__(self, set_images: List[np.array], set_masks: List[np.array]):
        """ 
        set_images: `List[np.array]`
            batch of numpy images (H,W,3)
        set_masks: `List[np.array]`
            batch of numpy masks (H,W)
        return: `Tuple(np.array, np.array)`
            new image and mask
        """

        result_image = np.zeros([self.height, self.width, 3], dtype=np.uint8)
        result_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        scale_x = self.scale_range[0] + random.random() * (self.scale_range[1] - self.scale_range[0])
        scale_y = self.scale_range[0] + random.random() * (self.scale_range[1] - self.scale_range[0])
        divid_point_x = int(scale_x * self.width)
        divid_point_y = int(scale_y * self.height)

        for i, (img, mask) in enumerate(zip(set_images, set_masks)):
            if i == 0:  # top-left
                img, mask = self.get_resize(img, mask, divid_point_x, divid_point_y)
                result_image[:divid_point_y, :divid_point_x, :] = img
                result_mask[:divid_point_y, :divid_point_x] = mask

            elif i == 1:  # top-right
                img, mask = self.get_resize(img, mask, self.width - divid_point_x, divid_point_y)
                result_image[:divid_point_y, divid_point_x:self.width, :] = img
                result_mask[:divid_point_y, divid_point_x:self.width] = mask
                
            elif i == 2:  # bottom-left
                img, mask = self.get_resize(img, mask, divid_point_x, self.height - divid_point_y)
                result_image[divid_point_y:self.height, :divid_point_x, :] = img
                result_mask[divid_point_y:self.height, :divid_point_x] = mask
              
            else:  # bottom-right
                img, mask = self.get_resize(img, mask, self.width - divid_point_x, self.height - divid_point_y)
                result_image[divid_point_y:self.height, divid_point_x:self.width, :] = img
                result_mask[divid_point_y:self.height, divid_point_x:self.width] = mask
            
        return result_image, result_mask