from torchvision.transforms.transforms import (ToTensor, Compose, RandomResizedCrop, 
                                                Resize, Normalize)
# from torchvision.transforms import RandAugment # torchvision 1.10
from . import TRANSFORM_REGISTRY

TRANSFORM_REGISTRY.register(RandomResizedCrop, prefix='TV')
TRANSFORM_REGISTRY.register(Normalize, prefix='TV')
TRANSFORM_REGISTRY.register(Resize, prefix='TV')
TRANSFORM_REGISTRY.register(Compose, prefix='TV')
TRANSFORM_REGISTRY.register(ToTensor, prefix='TV')
# TRANSFORM_REGISTRY.register(RandAugment, prefix='TV')