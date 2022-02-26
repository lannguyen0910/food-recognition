from .tta import TTA
from theseus.base.augmentations import TRANSFORM_REGISTRY

TRANSFORM_REGISTRY.register(TTA, prefix='Base')
