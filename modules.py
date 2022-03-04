from re import I
import cv2
from PIL import Image
import numpy as np
import os
import pandas as pd
from theseus.utilities.visualization.visualizer import Visualizer
from theseus.utilities.download import download_from_drive
from theseus.utilities import box_fusion, change_box_order
from theseus.apis.inference import *

from analyzer import get_info_from_db

CACHE_DIR = './weights'
CSV_FOLDER = './static/csv'
METADATA_FOLDER = './static/metadata'


class DetectionArguments:
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


weight_urls = {
    'yolov5s': "1rISMag8OCM5v99TYuavAobm3LkwjtAi9",
    "yolov5m": "1I649VGqkam_IcCCW8WUA965vPrW_pqDX",
    "yolov5l": "1sBciFcRav2ZE6jzhWnca9uegjQ4860om",
    "yolov5x": "1CRD6T9QtH9XEa-h985_Ho6jgLWu58zn0",
    "effnetb4": "1-K_iDfuhxQFHIF9HTy8SvfnIFwjqxtaX",
    "semantic_seg": "19JRQr9xs2SIeTxX0TQ0k4U9ZnihahvqC"
}


def download_pretrained_weights(name, output=None):
    return download_from_drive(weight_urls[name], output)


def draw_image(out_path, visualizer, result_dict, class_names):
    if os.path.isfile(out_path):
        os.remove(out_path)

    if "names" in result_dict.keys():
        visualizer.draw_bboxes_v2(
            out_path,
            result_dict["boxes"],
            result_dict["labels"],
            result_dict["scores"],
            label_names=result_dict["names"])
    else:
        visualizer.draw_bboxes_v2(
            out_path,
            result_dict["boxes"],
            result_dict["labels"],
            result_dict["scores"],
            obj_list=class_names)


def save_cache(result_dict, cache_name, cache_dir=CACHE_DIR, exclude=[]):
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

    for key in result_dict.keys():
        if key != 'boxes' and key not in exclude:
            cache_dict[key] = result_dict[key]
    df = pd.DataFrame(cache_dict)

    df.to_csv(f'{cache_dir}/{cache_name}.csv', index=False)


def check_cache(cache_name):
    return os.path.isfile(f'./{CACHE_DIR}/{cache_name}.csv')


def load_cache(image_name):
    df = pd.read_csv(f'./{CACHE_DIR}/{image_name}.csv')
    result_dict = {
        'boxes': [],
        'labels': [],
        'scores': []
    }

    ann = [i for i in zip(df.x, df.y, df.w, df.h, df.labels, df.scores)]

    for row in ann:
        x, y, w, h, label, score = row
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)
        box = [x, y, w, h]
        label = int(label)
        score = float(score)
        result_dict['boxes'].append(box)
        result_dict['labels'].append(label)
        result_dict['scores'].append(score)

    return result_dict


def drop_duplicate_fill0(result_dict):
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
    food_labels = food_dict['labels']
    food_names = [' '.join(class_names[int(i)].split('-'))
                  for i in food_labels]
    food_dict['names'] = food_names
    return food_dict


def append_food_info(food_dict):
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
    # expand box a little
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


def label_enhancement(image, result_dict):
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
        if label == 34 or label == 65:  # perform classification on "food" label and "food-drinks" label
            cropped = crop_box(image, box)  # rgb
            img_list.append(cropped.copy())
            new_id_list.append(box_id)

    tmp_path = os.path.join(CACHE_DIR, 'effnetb4.pth')
    if not os.path.isfile(tmp_path):
        download_pretrained_weights(
            'effnetb4',
            output=tmp_path)

    new_dict = classify(tmp_path, img_list)

    """
    new_dict = {
        'filename': [],
        'label': [],
        'score': []
    }
    """

    for idx, id in enumerate(new_id_list):
        result_dict['names'][id] = new_dict['label'][idx]

    return result_dict


def ensemble_models(input_path, image_size, tta=False):

    args1 = Arguments(model_name='yolov5s')
    args2 = Arguments(model_name='yolov5m')
    args3 = Arguments(model_name='yolov5l')
    args4 = Arguments(model_name='yolov5x')

    args1.input_path = input_path
    args2.input_path = input_path
    args3.input_path = input_path
    args4.input_path = input_path

    args1.tta = tta
    args2.tta = tta
    args3.tta = tta
    args4.tta = tta

    # class_names, num_classes = get_class_names(args1.weight)
    config1 = get_config(model_name='yolov5s')
    config2 = get_config(model_name='yolov5m')
    config3 = get_config(model_name='yolov5l')
    config4 = get_config(model_name='yolov5x')

    class_names, num_classes = get_class_names('yolov5s')

    result_dict1 = detect(args1, config1)
    result_dict2 = detect(args2, config2)
    result_dict3 = detect(args3, config3)
    result_dict4 = detect(args4, config4)

    merged_boxes = [
        np.array(result_dict1['boxes']),
        np.array(result_dict2['boxes']),
        np.array(result_dict3['boxes']),
        np.array(result_dict4['boxes'])]
    merged_labels = [
        np.array(result_dict1['labels']),
        np.array(result_dict2['labels']),
        np.array(result_dict3['labels']),
        np.array(result_dict4['labels'])]
    merged_scores = [
        np.array(result_dict1['scores']),
        np.array(result_dict2['scores']),
        np.array(result_dict3['scores']),
        np.array(result_dict4['scores'])]

    for i in range(len(merged_boxes)):
        merged_boxes[i][:, 2] += merged_boxes[i][:, 0]  # xyxy
        merged_boxes[i][:, 3] += merged_boxes[i][:, 1]  # xyxy

    final_boxes, final_scores, final_classes = box_fusion(
        merged_boxes,
        merged_scores,
        merged_labels,
        mode="wbf",
        image_size=image_size,
        iou_threshold=0.9,
        weights=[0.25, 0.25, 0.25, 0.25]
    )

    indexes = np.where(final_scores > 0.001)[0]
    final_boxes = final_boxes[indexes]
    final_scores = final_scores[indexes]
    final_classes = final_classes[indexes]

    final_boxes = change_box_order(final_boxes, order='xyxy2xywh')

    result_dict = {
        'boxes': [],
        'labels': [],
        'scores': []
    }

    for (box, score, label) in zip(final_boxes, final_scores, final_classes):
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

    # get hashed key from image path
    ori_hashed_key = os.path.splitext(os.path.basename(input_path))[0]

    if segmentation:
        seg_args = InferenceArguments(key="segmentation")
        opts = Opts(seg_args.config).parse_args()
        seg_pipeline = SegmentationPipeline(opts, input_path)
        seg_pipeline.inference()

        return output_path

    # additional tags
    model_tag = model_name[-1]
    ensemble_tag = 'ens' if ensemble else ''

    if ensemble:
        hashed_key = ori_hashed_key + f"_{ensemble_tag}"
    else:
        hashed_key = ori_hashed_key + f"_{model_tag}"

    ori_img = cv2.imread(input_path)

    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = ori_img.shape

    # check whether cache exists
    if check_cache(hashed_key):
        print(f"Load cache from {hashed_key}")
        # class_names, _ = get_class_names(model_name)
        result_dict = load_cache(hashed_key)

    else:
        if not ensemble:
            args = DetectionArguments(
                model_name=model_name,
                input_path=input_path,
                output_path=output_path,
                min_conf=min_conf,
                min_iou=min_iou,
                tta=tta,
                tta_ensemble_mode='wbf',
                tta_conf_threshold=0.01,
                tta_iou_threshold=0.9,
            )

            det_args = InferenceArguments(key="detection")
            opts = Opts(det_args.config).parse_args()
            det_pipeline = DetectionPipeline(opts, args)
            result_dict = det_pipeline.inference()

        else:
            result_dict, class_names = ensemble_models(
                input_path, [img_w, img_h], tta=tta)

        save_cache(result_dict, hashed_key)
        print(f"Save cache to {hashed_key}")

    # class_names.insert(0, "Background")

    # post process
    # result_dict = postprocess(result_dict, img_w, img_h, min_iou, min_conf)

    # add food name
    result_dict = append_food_name(result_dict, class_names)

    # enhance by using a classifier
    if enhance_labels:
        result_dict = label_enhancement(ori_img, result_dict)

    # add food infomation and save to file
    result_dict = append_food_info(result_dict)

    # Save metadata food info as CSV
    save_cache(result_dict, ori_hashed_key+'_metadata', METADATA_FOLDER)

    visualizer = Visualizer()
    visualizer.set_image(ori_img)

    # draw result
    draw_image(output_path, visualizer, result_dict, class_names)

    result_list = convert_dict_to_list(result_dict)

    # Save food info as CSV
    csv_result_dict = drop_duplicate_fill0(result_dict)
    save_cache(csv_result_dict, ori_hashed_key+'_info',
               CSV_FOLDER, exclude=['boxes', "labels", "scores"])

    # Transpose CSV
    df = pd.read_csv(os.path.join(CSV_FOLDER, ori_hashed_key+'_info.csv'))
    df.set_index('names').T.to_csv(os.path.join(
        CSV_FOLDER, ori_hashed_key+'_info2.csv'))

    return output_path
