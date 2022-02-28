import cv2
import albumentations as A


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
