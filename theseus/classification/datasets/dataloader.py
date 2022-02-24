import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler

def class_imbalance_sampler(labels: torch.Tensor):
    r""" Create balance sampler based on label distribution
    
    labels: `torch.Tensor`
        labels distribution
    """
    class_count = torch.bincount(labels.squeeze())
    class_weighting = 1. / class_count
    sample_weights = np.array([class_weighting[t] for t in labels.squeeze()])
    sample_weights = torch.from_numpy(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler

class BalanceSampler(torch.utils.data.DataLoader):
    r"""Balance DataLoader, equally distribute labels in one batch
    
    dataset: `torch.utils.data.Dataset`
        dataset, must have classes_dict and collate_fn attributes
    batch_size: `int`
        number of samples in one batch
    train: `bool`
        whether the dataloader is used for training or test
    """
    def __init__(self, 
        dataset: torch.utils.data.Dataset, 
        batch_size: int, 
        train: bool = True, 
        **kwargs):

        assert hasattr(dataset, 'classes_dist'), 'Dataset must define classes distribution'
        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        else:
            collate_fn = None

        # Only balancing for training dataloader    
        if train:
            labels = torch.LongTensor(dataset.classes_dist).unsqueeze(1)
            sampler = class_imbalance_sampler(labels)
        else:
            sampler = None
            
        super(BalanceSampler, self).__init__(
            dataset,
            batch_size=batch_size,
            collate_fn = collate_fn,
            sampler=sampler,
            **kwargs
        )