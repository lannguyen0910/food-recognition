from theseus.base.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from .csv_dataset import *
from .folder_dataset import *

DATASET_REGISTRY.register(CSVDataset)
DATASET_REGISTRY.register(ImageFolderDataset)