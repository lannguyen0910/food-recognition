from .tta import *
from .transforms import *
from theseus.base.augmentations import TRANSFORM_REGISTRY

TRANSFORM_REGISTRY.register(TTAHorizontalFlip)
TRANSFORM_REGISTRY.register(TTARotate90)
TRANSFORM_REGISTRY.register(TTAVerticalFlip)
TRANSFORM_REGISTRY.register(TTACompose)
TRANSFORM_REGISTRY.register(get_resize_augmentation)
