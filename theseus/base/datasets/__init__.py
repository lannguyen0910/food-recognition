from theseus.registry import Registry
from torch.utils.data import DataLoader, Dataset
from .dataloader import DataLoaderWithCollator
from .dataset import ChainDataset, ConcatDataset, ImageDataset
from .sampler import BalanceSampler
from .collator import ChainCollatorWrapper

DATASET_REGISTRY = Registry('DATASET')
DATASET_REGISTRY.register(Dataset)
DATASET_REGISTRY.register(ChainDataset)
DATASET_REGISTRY.register(ConcatDataset)
DATASET_REGISTRY.register(ImageDataset)

DATALOADER_REGISTRY = Registry('DATALOADER')
DATALOADER_REGISTRY.register(DataLoader)
DATALOADER_REGISTRY.register(BalanceSampler)
DATALOADER_REGISTRY.register(ChainCollatorWrapper)
DATALOADER_REGISTRY.register(DataLoaderWithCollator)