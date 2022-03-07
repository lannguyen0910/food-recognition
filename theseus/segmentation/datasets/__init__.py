from theseus.base.datasets import DATASET_REGISTRY, DATALOADER_REGISTRY

from .csv_dataset import CSVDataset
DATASET_REGISTRY.register(CSVDataset, "SEG_")

from .mosaic_collator import MosaicCollator
DATALOADER_REGISTRY.register(MosaicCollator)