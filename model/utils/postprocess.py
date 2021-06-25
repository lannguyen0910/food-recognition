import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
import webcolors
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ensemble_boxes import weighted_boxes_fusion, nms

def change_box_order(boxes, order):
    """
    Change box order between (xmin, ymin, xmax, ymax) and (xcenter, ycenter, width, height).
    :param boxes: (tensor) or {np.array) bounding boxes, sized [N, 4]
    :param order: (str) ['xyxy2xywh', 'xywh2xyxy', 'xyxy2cxcy', 'cxcy2xyxy']
    :return: (tensor) converted bounding boxes, size [N, 4]
    """

    assert order in ['xyxy2xywh', 'xywh2xyxy', 'xyxy2cxcy', 'cxcy2xyxy', 'yxyx2xyxy', 'xyxy2yxyx']

    # Convert 1-d to a 2-d tensor of boxes, which first dim is 1
    if isinstance(boxes, torch.Tensor):
        if len(boxes.shape) == 1:
            boxes = boxes.unsqueeze(0)

        if order == 'xyxy2xywh':
            return torch.cat([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], 1)
        elif order ==  'xywh2xyxy':
            return torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]], 1)
        elif order == 'xyxy2cxcy':
            return torch.cat([(boxes[:, 2:] + boxes[:, :2]) / 2,  # c_x, c_y
                            boxes[:, 2:] - boxes[:, :2]], 1)  # w, h
        elif order == 'cxcy2xyxy':
            return torch.cat([boxes[:, :2] - (boxes[:, 2:] *1.0 / 2),  # x_min, y_min
                            boxes[:, :2] + (boxes[:, 2:] *1.0 / 2)], 1)  # x_max, y_max
        elif order == 'xyxy2yxyx' or order == 'yxyx2xyxy':
            return boxes[:,[1,0,3,2]]
        
    else:
        # Numpy
        new_boxes = boxes.copy()
        if order == 'xywh2xyxy':
            new_boxes[:,2] = boxes[:,0] + boxes[:,2]
            new_boxes[:,3] = boxes[:,1] + boxes[:,3]
            return new_boxes
        elif order == 'xyxy2xywh':
            new_boxes[:,2] = boxes[:,2] - boxes[:,0]
            new_boxes[:,3] = boxes[:,3] - boxes[:,1]
            return new_boxes

def filter_area(boxes, labels, confidence_score=None, min_wh=10, max_wh=4096):
    """
    Boxes in xyxy format
    """

    # dimension of bounding boxes
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]

    width = width.astype(int)
    height = height.astype(int)

    picked_index_min = (width >= min_wh) & (height >= min_wh)
    picked_index_max = (width <= max_wh) & (height <= max_wh)

    picked_index = picked_index_min & picked_index_max

    # Picked bounding boxes
    picked_boxes = boxes[picked_index]
    picked_classes = labels[picked_index]
    if confidence_score is not None:
        picked_score = confidence_score[picked_index]
    
    if confidence_score is not None:
        return np.array(picked_boxes), np.array(picked_score), np.array(picked_classes)
    else:
        return np.array(picked_boxes), np.array(picked_classes)

def resize_postprocessing(boxes, current_img_size, ori_img_size, keep_ratio=False):
    """
    Boxes format must be in xyxy
    if keeping ratio, padding will be calculated then substracted from bboxes
    """

    new_boxes = boxes.copy()
    if keep_ratio:
        ori_w, ori_h = ori_img_size
        ratio = float(ori_w*1.0/ori_h)
        
        # If ratio equals 1.0, skip to scaling
        if ratio != 1.0: 
            if ratio > 1.0: # width > height, width = current_img_size, meaning padding along height
                true_width = current_img_size[0]
                true_height = current_img_size[0] / ratio # true height without padding equals (current width / ratio)
                pad_size = int((true_width-true_height)/2) # Albumentation padding
                
                # Subtract padding size from heights
                new_boxes[:,1] -= pad_size
                new_boxes[:,3] -= pad_size
            else: # height > width, height = current_img_size, meaning padding along width
                true_height = current_img_size[1]
                true_width = current_img_size[1] * ratio # true width without padding equals (current height * ratio)
                pad_size = int((true_height-true_width)/2) # Albumentation padding

                # Subtract padding size from widths
                new_boxes[:,0] -= pad_size
                new_boxes[:,2] -= pad_size
            # Assign new width, new height
            current_img_size = [true_width, true_height]
    
    # Scaling boxes to match original image shape 
    new_boxes[:,0] = (new_boxes[:,0] * ori_img_size[0])/ current_img_size[0]
    new_boxes[:,2] = (new_boxes[:,2] * ori_img_size[0])/ current_img_size[0]
    new_boxes[:,1] = (new_boxes[:,1] * ori_img_size[1])/ current_img_size[1]
    new_boxes[:,3] = (new_boxes[:,3] * ori_img_size[1])/ current_img_size[1]
    return new_boxes

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (width, height)
    if isinstance(boxes, torch.Tensor):
        _boxes = boxes.clone()
        _boxes[:, 0].clamp_(0, img_shape[0])  # x1
        _boxes[:, 1].clamp_(0, img_shape[1])  # y1
        _boxes[:, 2].clamp_(0, img_shape[0])  # x2
        _boxes[:, 3].clamp_(0, img_shape[1])  # y2
    else:
        _boxes = boxes.copy()
        _boxes[:, 0] = np.clip(_boxes[:, 0], 0, img_shape[0])  # x1
        _boxes[:, 1] = np.clip(_boxes[:, 1], 0, img_shape[1])  # y1
        _boxes[:, 2] = np.clip(_boxes[:, 2], 0, img_shape[0])  # x2
        _boxes[:, 3] = np.clip(_boxes[:, 3], 0, img_shape[1])  # y2

    return _boxes

def postprocessing(
        preds, 
        current_img_size=None,  # Need to be square
        ori_img_size=None,
        min_iou=0.5, 
        min_conf=0.1,
        mode=None,
        max_dets=None,
        keep_ratio=False,
        output_format='xywh'):
    """
    Input: bounding boxes in xyxy format
    Output: bounding boxes in xywh format
    """
    boxes, scores, labels = preds['bboxes'], preds['scores'], preds['classes']

    if len(boxes) == 0 or boxes is None:
        return {
            'bboxes': boxes, 
            'scores': scores, 
            'classes': labels}

    # Clip boxes in image size
    boxes = clip_coords(boxes, current_img_size)

    # Filter small area boxes
    boxes, scores, labels = filter_area(
        boxes, labels, scores, min_wh=2, max_wh=4096
    )

    current_img_size = current_img_size if current_img_size is not None else None
    if len(boxes) != 0:
        if mode is not None:
            boxes, scores, labels = box_fusion(
                [boxes],
                [scores],
                [labels],
                image_size=current_img_size[0],
                mode=mode,
                iou_threshold=min_iou)

        indexes = np.where(scores > min_conf)[0]
        
        boxes = boxes[indexes]
        scores = scores[indexes]
        labels = labels[indexes]

        if max_dets is not None:
            sorted_index = np.argsort(scores)
            boxes = boxes[sorted_index]
            scores = scores[sorted_index]
            labels = labels[sorted_index]
            
            boxes = boxes[:max_dets]
            scores = scores[:max_dets]
            labels = labels[:max_dets]

        if ori_img_size is not None and current_img_size is not None:
            boxes = resize_postprocessing(
                boxes, 
                current_img_size=current_img_size, 
                ori_img_size=ori_img_size, 
                keep_ratio=keep_ratio)

        if output_format == 'xywh':
            boxes = change_box_order(boxes, order='xyxy2xywh')


    return {
        'bboxes': boxes, 
        'scores': scores, 
        'classes': labels}

def box_fusion(
    bounding_boxes, 
    confidence_score, 
    labels, 
    mode='wbf', 
    image_size=None,
    weights=None, 
    iou_threshold=0.5):
    """
    bounding boxes: 
        list of boxes of same image [[box1, box2,...],[...]] if ensemble many models
        list of boxes of single image [[box1, box2,...]] if done on one model
    """

    if image_size is not None:
        boxes = [i*1.0/image_size for i in bounding_boxes]
    else:
        boxes = bounding_boxes.copy()

    if mode == 'wbf':
        picked_boxes, picked_score, picked_classes = weighted_boxes_fusion(
            boxes, 
            confidence_score, 
            labels, 
            weights=weights, 
            iou_thr=iou_threshold, 
            conf_type='avg', #[nms|avf]
            skip_box_thr=0.0001)
    elif mode == 'nms':
        picked_boxes, picked_score, picked_classes = nms(
            boxes, 
            confidence_score, 
            labels,
            weights=weights,
            iou_thr=iou_threshold)

    if image_size is not None:
        picked_boxes = picked_boxes*image_size

    return np.array(picked_boxes), np.array(picked_score), np.array(picked_classes)
