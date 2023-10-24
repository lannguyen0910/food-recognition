import numpy as np
import torch
from theseus.utilities.postprocess import box_fusion


class BaseTTA:
    """ author: @shonenkov """
    image_size = None

    def augment(self, image):
        raise NotImplementedError

    def batch_augment(self, images):
        raise NotImplementedError

    def deaugment_boxes(self, boxes):
        raise NotImplementedError


class TTAHorizontalFlip(BaseTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image.flip(1)

    def batch_augment(self, images):
        return images.flip(2)

    def deaugment_boxes(self, boxes):
        boxes[:, [1, 3]] = self.image_size - boxes[:, [3, 1]]
        return boxes


class TTAVerticalFlip(BaseTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image.flip(2)

    def batch_augment(self, images):
        return images.flip(3)

    def deaugment_boxes(self, boxes):
        boxes[:, [0, 2]] = self.image_size - boxes[:, [2, 0]]
        return boxes


class TTARotate90(BaseTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return torch.rot90(image, 1, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 1, (2, 3))

    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0, 2]] = self.image_size - boxes[:, [3, 1]]
        res_boxes[:, [1, 3]] = boxes[:, [0, 2]]
        return res_boxes


class TTACompose(BaseTTA):
    """ author: @shonenkov """

    def __init__(self, transforms):
        self.transforms = transforms

    def augment(self, image):
        for transform in self.transforms:
            image = transform.augment(image)
        return image

    def batch_augment(self, images):
        for transform in self.transforms:
            images = transform.batch_augment(images)
        return images

    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:, 0] = np.min(boxes[:, [0, 2]], axis=1)
        result_boxes[:, 2] = np.max(boxes[:, [0, 2]], axis=1)
        result_boxes[:, 1] = np.min(boxes[:, [1, 3]], axis=1)
        result_boxes[:, 3] = np.max(boxes[:, [1, 3]], axis=1)
        return result_boxes

    def deaugment_boxes(self, boxes):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)


class TTA():
    """
    Only support square images
    """

    def __init__(self, postprocess_mode='wbf', min_conf=0.1, min_iou=0.5):
        self.postprocess_mode = postprocess_mode
        self.min_conf = min_conf
        self.min_iou = min_iou
        self.postprocess_fn = box_fusion

        # self.tta_transforms = []
        # for tta_combination in product([TTAHorizontalFlip(), None],
        #                             [TTAVerticalFlip(), None],
        #                             [TTARotate90(), None]):
        #     self.tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))
        self.tta_transforms = [TTACompose([i]) if i is not None else None for i in [
            None, TTAHorizontalFlip(), TTAVerticalFlip(), TTARotate90()]]

    def make_tta_predictions(self, model, batch, device, weights=None):

        # Set image size for all transforms
        batch_size = 1
        image_size = batch['image_ori_ws'][0]  # [480, 640]
        for tta_transform in self.tta_transforms:
            if tta_transform is not None:
                for single_transform in tta_transform.transforms:
                    single_transform.image_size = int(image_size)

        final_outputs = []
        with torch.no_grad():
            predictions = {
                'bboxes': {},
                'classes': {},
                'scores': {}
            }
            for aug_idx, tta_transform in enumerate(self.tta_transforms):
                imgs = batch['torch_inputs']
                if tta_transform is not None:
                    tta_imgs = tta_transform.batch_augment(imgs)
                else:
                    tta_imgs = imgs

                tta_batch = {
                    'inputs': tta_imgs
                }

                #  Feed imgs through model
                outputs = model.get_prediction(tta_batch, device, is_tta=True)

                for i, output in enumerate(outputs):

                    if i not in predictions['bboxes'].keys():
                        predictions['bboxes'][i] = []
                        predictions['classes'][i] = []
                        predictions['scores'][i] = []

                    boxes = output['bboxes']
                    scores = output['scores']
                    classes = output['classes']

                    indexes = np.where(scores > self.min_conf)[0]

                    boxes = boxes[indexes]
                    scores = scores[indexes]
                    classes = classes[indexes]

                    if len(boxes) != 0:
                        if tta_transform is not None:
                            boxes = tta_transform.deaugment_boxes(boxes.copy())

                    predictions['bboxes'][i].append(boxes)
                    predictions['classes'][i].append(classes)
                    predictions['scores'][i].append(scores)

        # Ensemble all boxes of each images
        for i in range(batch_size):

            final_boxes, final_scores, final_classes = self.postprocess_fn(
                predictions['bboxes'][i],
                predictions['scores'][i],
                predictions['classes'][i],
                mode=self.postprocess_mode,
                image_size=image_size,
                iou_threshold=self.min_iou,
            )

            indexes = np.where(final_scores > self.min_conf)[0]
            final_boxes = final_boxes[indexes]
            final_scores = final_scores[indexes]
            final_classes = final_classes[indexes]

            final_outputs.append({
                'bboxes': final_boxes,
                'scores': final_scores,
                'classes': final_classes,
            })
        return final_outputs
