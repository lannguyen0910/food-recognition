from theseus.base.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from .csv_dataset import CSVDataset
from .mosaic_dataset import CSVDatasetWithMosaic

DATASET_REGISTRY.register(CSVDataset)
DATASET_REGISTRY.register(CSVDatasetWithMosaic)
