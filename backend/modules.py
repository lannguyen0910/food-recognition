import os
import cv2
import numpy as np
import pandas as pd

from theseus.utilities.visualization.utils import draw_bboxes_v2
from theseus.utilities.download import download_pretrained_weights
from theseus.utilities import box_fusion, postprocessing
from theseus.apis.inference import SegmentationPipeline, DetectionPipeline, ClassificationPipeline
from theseus.opt import Opts, InferenceArguments

from .edamam.api import get_info_from_db
from .constants import CACHE_DIR, CSV_FOLDER


class DetectionArguments:
    """
    Arguments from input to perform food detection
    """
    def __init__(
        self,
        model_name: str = None,
        input_path: str = "",
        output_path: str = "",
        min_conf: float = 0.001,
        min_iou: float = 0.99,
        tta: bool = False,
        tta_ensemble_mode: str = 'wbf',
        tta_conf_threshold: float = 0.01,
        tta_iou_threshold: float = 0.9,
    ) -> None:
        self.model_name = model_name
        self.weight = None
        self.input_path = input_path
        self.output_path = output_path
        self.min_conf = min_conf
        self.min_iou = min_iou
        self.tta = tta
        self.tta_ensemble_mode = tta_ensemble_mode
        self.tta_conf_threshold = tta_conf_threshold
        self.tta_iou_threshold = tta_iou_threshold

        if self.model_name:
            tmp_path = os.path.join(CACHE_DIR, self.model_name+'.pt')
            download_pretrained_weights(
                self.model_name,
                output=tmp_path)
            self.weight = tmp_path


def draw_image(out_path, img, result_dict, class_names):
    """
    Draw bboxes and labels for detected image
    """
    if os.path.isfile(out_path):
        os.remove(out_path)

    if "names" in result_dict.keys():
        draw_bboxes_v2(
            out_path,
            img,
            result_dict["boxes"],
            result_dict["labels"],
            result_dict["scores"],
            label_names=result_dict["names"])
    else:
        draw_bboxes_v2(
            out_path,
            img,
            result_dict["boxes"],
            result_dict["labels"],
            result_dict["scores"],
            obj_list=class_names)


def save_cache(result_dict, cache_name, cache_dir=CACHE_DIR, exclude=[]):
    """
    Save detection info to csv
    """
    cache_dict = {}
    if 'boxes' not in exclude:
        boxes = np.array(result_dict['boxes'])
        if len(boxes) != 0:
            cache_dict.update({
                'x': boxes[:, 0],
                'y': boxes[:, 1],
                'w': boxes[:, 2],
                'h': boxes[:, 3],
            })

    for key in cache_dict.keys():
        if len(cache_dict[key]) == 0:
            return

    for key in result_dict.keys():
        if key != 'boxes' and key not in exclude:
            cache_dict[key] = result_dict[key]

    df = pd.DataFrame(cache_dict)

    df.to_csv(f'{cache_dir}/{cache_name}.csv', index=False)


def drop_duplicate_fill0(result_dict):
    """
    Drop value-0 from detection result
    """
    labels = result_dict['labels']
    num_items = len(labels)

    label_set = set()
    keep_index = []
    for i in range(num_items):
        if labels[i] not in label_set:
            label_set.add(labels[i])
            keep_index.append(i)

    new_result_dict = {}

    for key in result_dict.keys():
        new_result_dict[key] = []
        for i in keep_index:
            value = result_dict[key][i]
            if value is None:
                value = 0
            new_result_dict[key].append(value)

    return new_result_dict


def append_food_name(food_dict, class_names):
    """
    Append food names from labels for nutrition analysis
    """
    food_labels = food_dict['labels']  # [0].to_list()
    food_names = [' '.join(class_names[int(i)].split('-'))
                  for i in food_labels]
    food_dict['names'] = food_names
    return food_dict


def append_food_info(food_dict):
    """
    Append nutrition info from database (db.json)
    """
    food_names = food_dict['names']
    food_info = get_info_from_db(food_names)
    food_dict.update(food_info)
    return food_dict


def convert_dict_to_list(result_dict):
    result_list = []
    num_items = len(result_dict['labels'])
    for i in range(num_items):
        item_dict = {}
        for key in result_dict.keys():
            item_dict[key] = result_dict[key][i]
        result_list.append(item_dict)
    return result_list


def crop_box(image, box, expand=10):
    h, w, c = image.shape

    # expand box a little (optional)
    new_box = box.copy()
    # new_box[0] -= expand
    # new_box[1] -= expand
    # new_box[2] += expand
    # new_box[3] += expand

    # new_box[0] = max(0, new_box[0])
    # new_box[1] = max(0, new_box[1])
    # new_box[2] = min(h, new_box[2])
    # new_box[3] = min(w, new_box[3])

    # xyxy box, cv2 image h,w,c
    return image[int(new_box[1]):int(new_box[3]), int(new_box[0]):int(new_box[2]), :]


def postprocess(result_dict, img_w, img_h, min_iou, min_conf):

    boxes = np.array(result_dict['boxes'])
    scores = np.array(result_dict['scores'])
    labels = np.array(result_dict['labels'])
    if len(boxes) != 0:
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        outputs = {
            'bboxes': boxes,
            'scores': scores,
            'classes': labels
        }

        outputs = postprocessing(
            outputs,
            current_img_size=[img_w, img_h],
            min_iou=min_iou,
            min_conf=min_conf,
            output_format='xywh',
            mode='nms')

        boxes = outputs['bboxes']
        labels = outputs['classes']
        scores = outputs['scores']

    return {
        'boxes': boxes,
        'labels': labels,
        'scores': scores
    }


def label_enhancement(image, result_dict):
    """
    Perform classification on cropped images (in some specific labels)
    """
    boxes = np.array(result_dict['boxes'])
    labels = np.array(result_dict['labels'])
    if len(boxes) == 0:
        return result_dict
    boxes[:, 2] += boxes[:, 0]  # xyxy
    boxes[:, 3] += boxes[:, 1]  # xyxy

    # Label starts at 1
    img_list = []
    new_id_list = []

    for box_id, (box, label) in enumerate(zip(boxes, labels)):
        if label == 34 or label == 65:  # classification on "food" label and "food-drinks" label
            cropped = crop_box(image, box)  # rgb
            img_list.append(cropped.copy())
            new_id_list.append(box_id)

    tmp_path = os.path.join(CACHE_DIR, 'effnetb4.pth')
    if not os.path.isfile(tmp_path):
        download_pretrained_weights(
            'effnetb4',
            output=tmp_path)

    cls_args = InferenceArguments(key="classification")
    opts = Opts(cls_args).parse_args()
    val_pipeline = ClassificationPipeline(opts, img_list)
    new_dict = val_pipeline.inference()

    for label_id, name_id in enumerate(new_id_list):
        result_dict['names'][name_id] = new_dict['label'][label_id]

    return result_dict


def ensemble_models(input_path, image_size, min_iou, min_conf, tta=False):
    """
    Ensemble technique on 4 YOLOv5 models
    """
    args1 = DetectionArguments(
        model_name='yolov5s', input_path=input_path, tta=tta)
    args2 = DetectionArguments(
        model_name='yolov5m', input_path=input_path, tta=tta)
    args3 = DetectionArguments(
        model_name='yolov5l', input_path=input_path, tta=tta)
    args4 = DetectionArguments(
        model_name='yolov5x', input_path=input_path, tta=tta)

    det_args = InferenceArguments(key="detection")
    opts = Opts(det_args).parse_args()

    det_pipeline1 = DetectionPipeline(opts, args1)
    result_dict1 = det_pipeline1.inference()

    det_pipeline2 = DetectionPipeline(opts, args2)
    result_dict2 = det_pipeline2.inference()

    det_pipeline3 = DetectionPipeline(opts, args3)
    result_dict3 = det_pipeline3.inference()

    det_pipeline4 = DetectionPipeline(opts, args4)
    result_dict4 = det_pipeline4.inference()
    class_names = det_pipeline4.class_names

    merged_boxes = [
        np.array(result_dict1['boxes'][0]),
        np.array(result_dict2['boxes'][0]),
        np.array(result_dict3['boxes'][0]),
        np.array(result_dict4['boxes'][0])]
    merged_labels = [
        np.array(result_dict1['labels'][0]),
        np.array(result_dict2['labels'][0]),
        np.array(result_dict3['labels'][0]),
        np.array(result_dict4['labels'][0])]
    merged_scores = [
        np.array(result_dict1['scores'][0]),
        np.array(result_dict2['scores'][0]),
        np.array(result_dict3['scores'][0]),
        np.array(result_dict4['scores'][0])]

    for i, _ in enumerate(merged_boxes):
        merged_boxes[i][:, 2] += merged_boxes[i][:, 0]  # xyxy
        merged_boxes[i][:, 3] += merged_boxes[i][:, 1]  # xyxy

    final_boxes, final_scores, final_classes = box_fusion(
        merged_boxes,
        merged_scores,
        merged_labels,
        mode="wbf",
        image_size=image_size,
        iou_threshold=0.9,
        # YOLOv5l performance best, YOLOv5x performance least (in current weights)
        weights=[0.25, 0.25, 0.4, 0.1]
    )

    final_dict = {
        'boxes': final_boxes,
        'labels': final_classes,
        'scores': final_scores
    }

    final_dict = postprocess(
        final_dict, image_size[1], image_size[0], min_iou, min_conf)

    # final_dict['boxes'] = change_box_order(final_dict['boxes'], order='xywh2xyxy')

    result_dict = {
        'boxes': [],
        'labels': [],
        'scores': []
    }

    for (box, score, label) in zip(final_dict['boxes'], final_dict['scores'], final_dict['labels']):
        result_dict['boxes'].append(box)
        result_dict['labels'].append(label)
        result_dict['scores'].append(score)

    return result_dict, class_names


def get_prediction(
        input_path,
        output_path,
        model_name,
        tta=False,
        ensemble=False,
        min_iou=0.5,
        min_conf=0.1,
        segmentation=False,
        enhance_labels=False):
    
    if segmentation:
        tmp_path = os.path.join(CACHE_DIR, 'semantic_seg.pth')
        if not os.path.isfile(tmp_path):
            download_pretrained_weights(
                'semantic_seg',
                output=tmp_path)

        seg_args = InferenceArguments(key="segmentation")
        opts = Opts(seg_args).parse_args()
        seg_pipeline = SegmentationPipeline(opts, input_path)
        output_path = seg_pipeline.inference()

        # get real output for segmentation task to display in webapp
        output_path = output_path.split('/')[-3:]
        output_path = os.path.join(
            output_path[0], output_path[1], output_path[2])

        return output_path, 'semantic'

    # get hashed key from image path
    ori_hashed_key = os.path.splitext(os.path.basename(input_path))[0]

    ori_img = cv2.imread(input_path)

    ori_img = np.array(ori_img, dtype=np.uint16)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = ori_img.shape

    if not ensemble:
        args = DetectionArguments(
            model_name=model_name,
            input_path=input_path,
            output_path=output_path,
            min_conf=min_conf,
            min_iou=min_iou,
            tta=tta
        )

        det_args = InferenceArguments(key="detection")
        opts = Opts(det_args).parse_args()
        det_pipeline = DetectionPipeline(opts, args)
        class_names = det_pipeline.class_names

        result_dict = det_pipeline.inference()

        result_dict['boxes'] = result_dict['boxes'][0]
        result_dict['labels'] = result_dict['labels'][0]
        result_dict['scores'] = result_dict['scores'][0]

        # Post process (optional)
        # result_dict = postprocess(result_dict, img_w, img_h, min_iou, min_conf)

    else:
        result_dict, class_names = ensemble_models(
            input_path, [img_w, img_h], min_iou, min_conf, tta=tta)

    # add food name
    result_dict = append_food_name(result_dict, class_names)

    # enhance by using a classifier
    if enhance_labels:
        result_dict = label_enhancement(ori_img, result_dict)

    # add food infomation and save to file
    result_dict = append_food_info(result_dict)

    # draw result
    draw_image(output_path, ori_img, result_dict, class_names)

    result_list = convert_dict_to_list(result_dict)

    # Save food info as CSV
    csv_result_dict = drop_duplicate_fill0(result_dict)
    save_cache(csv_result_dict, ori_hashed_key+'_info',
               CSV_FOLDER, exclude=['boxes', "labels", "scores"])

    # Transpose CSV
    df = pd.read_csv(os.path.join(CSV_FOLDER, ori_hashed_key+'_info.csv'))
    df.set_index('names').T.to_csv(os.path.join(
        CSV_FOLDER, ori_hashed_key+'_info2.csv'))

    return output_path, 'detection'
