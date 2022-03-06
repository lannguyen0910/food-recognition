from albumentations import (Compose, Normalize, RandomBrightnessContrast,
                            RandomCrop, Resize, RGBShift, ShiftScaleRotate,
                            SmallestMaxSize, MotionBlur, GaussianBlur, MedianBlur,
                            Blur, RandomRotate90, HorizontalFlip, VerticalFlip,
                            HueSaturationValue, RandomSizedCrop, IAASharpen, BboxParams)
from albumentations.pytorch.transforms import ToTensorV2

from . import TRANSFORM_REGISTRY

TRANSFORM_REGISTRY.register(RandomCrop, prefix='Alb')
TRANSFORM_REGISTRY.register(RGBShift, prefix='Alb')
TRANSFORM_REGISTRY.register(Normalize, prefix='Alb')
TRANSFORM_REGISTRY.register(Resize, prefix='Alb')
TRANSFORM_REGISTRY.register(Compose, prefix='Alb')
TRANSFORM_REGISTRY.register(RandomBrightnessContrast, prefix='Alb')
TRANSFORM_REGISTRY.register(ShiftScaleRotate, prefix='Alb')
TRANSFORM_REGISTRY.register(SmallestMaxSize, prefix='Alb')
TRANSFORM_REGISTRY.register(MotionBlur, prefix='Alb')
TRANSFORM_REGISTRY.register(GaussianBlur, prefix='Alb')
TRANSFORM_REGISTRY.register(MedianBlur, prefix='Alb')
TRANSFORM_REGISTRY.register(Blur, prefix='Alb')
TRANSFORM_REGISTRY.register(RandomRotate90, prefix='Alb')
TRANSFORM_REGISTRY.register(HorizontalFlip, prefix='Alb')
TRANSFORM_REGISTRY.register(VerticalFlip, prefix='Alb')
TRANSFORM_REGISTRY.register(HueSaturationValue, prefix='Alb')
TRANSFORM_REGISTRY.register(RandomSizedCrop, prefix='Alb')
TRANSFORM_REGISTRY.register(IAASharpen, prefix='Alb')
TRANSFORM_REGISTRY.register(ToTensorV2, prefix='Alb')
TRANSFORM_REGISTRY.register(BboxParams, prefix='Alb')