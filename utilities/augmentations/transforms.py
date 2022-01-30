import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from .custom import CustomCutout
from utilities.configs import Config

# FOR BEST RESULTS, CHOOSE THE APPRORIATE NUMBERS
# If use EfficientDet, use these numbers
# MEAN = [0.485, 0.456, 0.406]
# STD = [0.229, 0.224, 0.225]

# If use YOLO, use these numbers
MEAN = [0.0, 0.0, 0.0]
STD = [1.0, 1.0, 1.0]


class Denormalize(object):
    """
    Denormalize image and boxes for visualization
    """

    def __init__(self, mean=MEAN, std=STD, **kwargs):
        self.mean = mean
        self.std = std

    def __call__(self, img, box=None, label=None, mask=None, **kwargs):
        """
        :param img: (tensor) image to be denormalized
        :param box: (list of tensor) bounding boxes to be denormalized, by multiplying them with image's width and heights. Format: (x,y,width,height)
        """
        mean = np.array(self.mean)
        std = np.array(self.std)
        img_show = img.numpy().squeeze().transpose((1, 2, 0))
        img_show = (img_show * std+mean)
        img_show = np.clip(img_show, 0, 1)
        return img_show


def get_resize_augmentation(image_size, keep_ratio=False, box_transforms=False):
    """
    Resize an image, support multi-scaling
    :param image_size: shape of image to resize
    :param keep_ratio: whether to keep image ratio
    :param box_transforms: whether to augment boxes
    :return: albumentation Compose
    """
    bbox_params = A.BboxParams(
        format='pascal_voc',
        min_area=0,
        min_visibility=0,
        label_fields=['class_labels']) if box_transforms else None

    if not keep_ratio:
        return A.Compose([
            A.Resize(
                height=image_size[1],
                width=image_size[0]
            )],
            bbox_params=bbox_params)
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=max(image_size)),
            A.PadIfNeeded(
                min_height=image_size[1], min_width=image_size[0], p=1.0, border_mode=cv2.BORDER_CONSTANT),
        ],
            bbox_params=bbox_params)


def get_augmentation(_type='train'):

    config = Config('./augmentations/augments.yaml')
    blur_config = config.blur
    flip_config = config.flip
    ssr_config = flip_config['shift_scale_crop']
    color_config = config.color
    removal_config = config.removal
    cutout_config = removal_config['cutout']

    transforms_list = [
        A.OneOf([
            A.MotionBlur(p=blur_config['motion']),
            A.GaussianBlur(p=blur_config['gaussian']),
            A.MedianBlur(blur_limit=3, p=blur_config['median']),
            A.Blur(blur_limit=3, p=blur_config['default']),
        ], p=blur_config['prob']),

        A.OneOf([
            A.RandomRotate90(p=flip_config['rotate90']),
            A.HorizontalFlip(p=flip_config['hflip']),
            A.VerticalFlip(p=flip_config['vflip']),
        ], p=flip_config['prob']),

        A.OneOf([
            A.HueSaturationValue(
                hue_shift_limit=color_config['hue'],
                sat_shift_limit=color_config['saturation'],
                val_shift_limit=color_config['value'],
                p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=color_config['brightness'],
                contrast_limit=color_config['contrast'],
                p=0.5)
        ], p=0.7),

        A.OneOf([
            A.IAASharpen(p=color_config['sharpen']),
            A.Compose([
                A.FromFloat(dtype='uint8', p=1),
                A.OneOf([
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(
                        8, 8), p=color_config['clahe']),
                    A.JpegCompression(p=color_config['compression']),
                ], p=0.7),
                A.ToFloat(p=1),
            ])
        ], p=color_config['prob']),

        # A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
        A.ShiftScaleRotate(
            shift_limit=ssr_config['shift_limit'],
            scale_limit=ssr_config['scale_limit'],
            rotate_limit=ssr_config['rotate_limit'],
            p=ssr_config['prob']),

        CustomCutout(
            bbox_removal_threshold=cutout_config['threshold'],
            min_cutout_size=cutout_config['min_size'],
            max_cutout_size=cutout_config['max_size'],
            number=cutout_config['number'],
            p=cutout_config['prob']),
    ]

    transforms_list += [
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=1.0, p=1.0),
        ToTensorV2(p=1.0)
    ]

    train_transforms = A.Compose(transforms_list, bbox_params=A.BboxParams(
        format='pascal_voc',
        min_area=2,
        min_visibility=0.2,
        label_fields=['class_labels']))

    val_transforms = A.Compose([
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=1.0, p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        min_area=0,
        min_visibility=0,
        label_fields=['class_labels']))

    return train_transforms if _type == 'train' else val_transforms
