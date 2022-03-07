from typing import List
import torch
from theseus.base.datasets.collator import BaseCollator
from theseus.classification.augmentations.custom import RandomMixup, RandomCutmix
import numpy as np

class MixupCutmixCollator(BaseCollator):
    """Apply mixup and cutmix to a batch, temporarily supports classification only
    """
    def __init__(
        self, 
        dataset: torch.utils.data.Dataset, 
        mixup_alpha: float=0.2, cutmix_alpha: float=1.0, 
        weight: List[float]=[0.5, 0.5], **kwargs) -> None:

        assert sum(weight) <= 1.0, "Sum of weight should be smaller than 1.0"
        self.mixup_transforms = []
        self.mixup_transforms.append(RandomMixup(dataset.num_classes, p=1.0, alpha=mixup_alpha))
        self.mixup_transforms.append(RandomCutmix(dataset.num_classes, p=1.0, alpha=cutmix_alpha))
        self.mixup_transforms.append(None)
        self.weight = weight
        self.weight.append(1.0-sum(weight))

    def __call__(self, batch):

        transform = np.random.choice(self.mixup_transforms, p=self.weight)
        if transform is not None:
            imgs, targets = transform(
                batch['inputs'], batch['targets'].squeeze(1))
            batch['inputs'] = imgs
            batch['targets'] = targets
        return batch