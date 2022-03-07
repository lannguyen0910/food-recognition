import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from theseus.utilities.loggers.observer import LoggerObserver

LOGGER = LoggerObserver.getLogger('main')

class BalanceSampler(WeightedRandomSampler):
    def __init__(self, dataset: torch.utils.data.Dataset, **kwargs):
        r""" Create balance sampler based on label distribution
        equally distribute labels in one batch

        dataset: `torch.utils.data.Dataset`
            dataset, must have classes_dict and collate_fn attributes

        **Note**:   the dataset must have `_calculate_classes_dist()` method 
                    that return `classes_dist` 
        """

        labels = self._load_labels(dataset)
        class_count = torch.bincount(labels.squeeze())
        class_weighting = 1. / class_count
        sample_weights = np.array([class_weighting[t] for t in labels.squeeze()])
        sample_weights = torch.from_numpy(sample_weights)
        super().__init__(sample_weights, len(sample_weights))

    def _load_labels(self, dataset):
        op = getattr(dataset, '_calculate_classes_dist', None)
        if not callable(op):
            LOGGER.text("""Using BalanceSampler but _calculate_classes_dist()
            method is missing from the dataset""", LoggerObserver.ERROR)
            raise ValueError

        classes_dist = dataset._calculate_classes_dist()
        labels = torch.LongTensor(classes_dist).unsqueeze(1)
        return labels