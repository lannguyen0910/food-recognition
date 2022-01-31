from re import I
import cv2
from PIL import Image
import numpy as np
import os
import pandas as pd
from utilities import (
    detect, Config, get_config, get_class_names,
    download_weights, draw_boxes_v2, postprocessing,
    box_fusion, classify, change_box_order,
    VideoPipeline)
from api import get_info_from_db

CACHE_DIR = '.cache'
CSV_FOLDER = './static/csv'
METADATA_FOLDER = './static/metadata'


class Arguments:
    def __init__(self, model_name=None) -> None:
        self.model_name = model_name
        self.weight = None
        self.input_path = ""
        self.output_path = ""
        self.gpus = "0"
        self.min_conf = 0.001
        self.min_iou = 0.99
        self.tta = False
        self.tta_ensemble_mode = "wbf"
        self.tta_conf_threshold = 0.01
        self.tta_iou_threshold = 0.9

        if self.model_name:
            tmp_path = os.path.join(CACHE_DIR, self.model_name+'.pt')
            download_pretrained_weights(
                self.model_name,
                cached=tmp_path)
            self.weight = tmp_path


weight_urls = {
    'yolov5s': "1rISMag8OCM5v99TYuavAobm3LkwjtAi9",
    "yolov5m": "1I649VGqkam_IcCCW8WUA965vPrW_pqDX",
    "yolov5l": "1sBciFcRav2ZE6jzhWnca9uegjQ4860om",
    "yolov5x": "1CRD6T9QtH9XEa-h985_Ho6jgLWu58zn0",
    "effnetb4": "1-K_iDfuhxQFHIF9HTy8SvfnIFwjqxtaX",
}


def download_pretrained_weights(name, cached=None):
    return download_weights(weight_urls[name], cached)


def draw_image(out_path, ori_img, result_dict, class_names):
    if os.path.isfile(out_path):
        os.remove(out_path)

    if "names" in result_dict.keys():
        draw_boxes_v2(
            out_path,
            ori_img,
            result_dict["boxes"],
            result_dict["labels"],
            result_dict["scores"],
            label_names=result_dict["names"])
    else:
        draw_boxes_v2(
            out_path,
            ori_img,
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


def ensemble_models(input_path, image_size):

    ignore_keys = [
        'min_iou_val',
        'min_conf_val',
        'tta',
        'gpu_devices',
        'tta_ensemble_mode',
        'tta_conf_threshold',
        'tta_iou_threshold',
    ]

    args1 = Arguments(model_name='yolov5s')
    args2 = Arguments(model_name='yolov5m')
    args3 = Arguments(model_name='yolov5l')
    args4 = Arguments(model_name='yolov5x')

    args1.input_path = input_path
    args2.input_path = input_path
    args3.input_path = input_path
    args4.input_path = input_path

    # class_names, num_classes = get_class_names(args1.weight)
    config1 = get_config(model_name='yolov5s')
    config2 = get_config(model_name='yolov5m')
    config3 = get_config(model_name='yolov5l')
    config4 = get_config(model_name='yolov5x')

    # config1 = Config(os.path.join('utilities', 'configs', 'yolov5s.yaml'))
    # config2 = Config(os.path.join('utilities', 'configs', 'yolov5m.yaml'))
    # config3 = Config(os.path.join('utilities', 'configs', 'yolov5l.yaml'))
    # config4 = Config(os.path.join('utilities', 'configs', 'yolov5x.yaml'))

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
        if label == 21:  # other food 31
            cropped = crop_box(image, box)  # rgb
            img_list.append(cropped.copy())
            new_id_list.append(box_id)

    tmp_path = os.path.join(CACHE_DIR, 'effnetb4.pth')
    if not os.path.isfile(tmp_path):
        download_pretrained_weights(
            'effnetb4',
            cached=tmp_path)

    new_names, new_probs = classify(tmp_path, img_list)

    for idx, id in enumerate(new_id_list):
        result_dict['names'][id] = new_names[idx]

    return result_dict


def get_video_prediction(
        input_path,
        output_path,
        model_name,
        min_iou=0.5,
        min_conf=0.1,
        enhance_labels=False):

    # ignore_keys = [
    #     'min_iou_val',
    #     'min_conf_val',
    #     'tta',
    #     'gpu_devices',
    #     'tta_ensemble_mode',
    #     'tta_conf_threshold',
    #     'tta_iou_threshold',
    # ]

    args = Arguments(model_name=model_name)

    config = get_config(model_name)

    if config is None:
        print("Config not found. Load configs from configs/configs.yaml")
        config = Config(os.path.join(
            'utilities/configs', 'train_detection.yaml'))
    else:
        print("Load configs from yaml file!")

    args.input_path = input_path
    args.output_path = output_path
    args.min_conf = min_conf
    args.min_iou = min_iou
    video_detect = VideoPipeline(args, config)
    return video_detect.run()


def get_prediction(
        input_path,
        output_path,
        model_name,
        ensemble=False,
        min_iou=0.5,
        min_conf=0.1,
        enhance_labels=False):

    # ignore_keys = [
    #     'min_iou_val',
    #     'min_conf_val',
    #     'tta',
    #     'gpu_devices',
    #     'tta_ensemble_mode',
    #     'tta_conf_threshold',
    #     'tta_iou_threshold',
    # ]

    # get hashed key from image path
    ori_hashed_key = os.path.splitext(os.path.basename(input_path))[0]

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
        class_names, _ = get_class_names(model_name)
        result_dict = load_cache(hashed_key)
    else:
        if not ensemble:
            args = Arguments(model_name=model_name)
            class_names, _ = get_class_names(model_name)

            config = get_config(model_name)
            if config is None:
                print("Config not found. Load configs from configs/configs.yaml")
                config = Config(os.path.join('model/configs', 'configs.yaml'))
            else:
                print("Load configs from weight")

            args.input_path = input_path
            result_dict = detect(args, config)

        else:
            result_dict, class_names = ensemble_models(
                input_path, [img_w, img_h])
        save_cache(result_dict, hashed_key)
        print(f"Save cache to {hashed_key}")

    print('Class names: ', class_names)
    class_names.insert(0, "Background")
    print('After insert: ', class_names)

    # post process
    result_dict = postprocess(result_dict, img_w, img_h, min_iou, min_conf)

    # add food name
    result_dict = append_food_name(result_dict, class_names)

    # enhance by using a classifier
    if enhance_labels:
        result_dict = label_enhancement(ori_img, result_dict)

    # add food infomation and save to file
    result_dict = append_food_info(result_dict)

    # Save metadata food info as CSV
    save_cache(result_dict, ori_hashed_key+'_metadata', METADATA_FOLDER)

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

    return output_path
