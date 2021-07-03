import cv2
from PIL import Image
import numpy as np
import os
import pandas as pd
from model import (
    detect, get_config, Config, 
    download_weights, draw_boxes_v2, 
    get_class_names, postprocessing, 
    box_fusion, classify)
from api import get_info_from_db

CACHE_DIR = '.cache'

class Arguments:
    def __init__(self, model_name=None) -> None:
        self.model_name = model_name
        self.weight = None
        self.input_path=""
        self.output_path=""
        self.gpus="0"
        self.min_conf=0.001
        self.min_iou=0.99
        self.tta=False
        self.tta_ensemble_mode="wbf"
        self.tta_conf_threshold=0.01
        self.tta_iou_threshold=0.9

        if self.model_name:
            tmp_path = os.path.join(CACHE_DIR, self.model_name+'.pth')
            download_pretrained_weights(
                self.model_name, 
                cached=tmp_path)
            self.weight=tmp_path
            
weight_urls = {
    'yolov5s': "1-3TXxsF_CYjPzQqEVudNUtMA5nexc8TG",
    'yolov5m': "1-EDbsoPOlYlkZGjol5sDSG4bhlJbgkDI",
    "yolov5l": "1-BfDjNXAjphIeJ0F1eJUborsHflwbeiI",
    "yolov5x": "1-5BSu6v9x9Dpdrya_o8RluzDV9aUSTgP",
    "effnetb4": "1-4AZSXhKAViZdM5PkhoeOZITVFM0WKIm",
    "yolov5m_extra": "1-HBTIM8pqXbppBiOHBVC9unWkUPjiZ_U"
}

def download_pretrained_weights(name, cached=None):
    return download_weights(weight_urls[name], cached)


def draw_image(out_path, ori_img, result_dict, class_names):
    if os.path.isfile(out_path):
        os.remove(out_path)

    if "names" in result_dict.keys():
        draw_boxes_v2(
            out_path, 
            ori_img , 
            result_dict["boxes"], 
            result_dict["labels"], 
            result_dict["scores"],
            label_names = result_dict["names"])
    else:
        draw_boxes_v2(
            out_path, 
            ori_img , 
            result_dict["boxes"], 
            result_dict["labels"], 
            result_dict["scores"],
            obj_list = class_names)


def save_cache(result_dict, cache_name):
    boxes = np.array(result_dict['boxes'])
   
    cache_dict = {
        'x': boxes[:, 0],
        'y': boxes[:, 1],
        'w': boxes[:, 2],
        'h': boxes[:, 3],
        'labels': result_dict['labels'],
        'scores': result_dict['scores'],
    }
    df = pd.DataFrame(cache_dict)

    df.to_csv(f'./{CACHE_DIR}/{cache_name}.csv', index=False)

def check_cache(cache_name):
    return os.path.isfile(f'./{CACHE_DIR}/{cache_name}.csv')

def load_cache(image_name):
    df = pd.read_csv(f'./{CACHE_DIR}/{image_name}.csv')
    result_dict = {
        'boxes': [],
        'labels': [],
        'scores': []
    }
    for idx, row in df.iterrows():
        x, y, w, h, label, score = row
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)
        box = [x,y,w,h]
        label = int(label)
        score = float(score)
        result_dict['boxes'].append(box)
        result_dict['labels'].append(label)
        result_dict['scores'].append(score)

    return result_dict

def postprocess(result_dict, img_w, img_h, min_iou, min_conf):
    
    boxes = np.array(result_dict['boxes'])
    scores = np.array(result_dict['scores'])
    labels = np.array(result_dict['labels'])
    boxes[:,2] += boxes[:,0] 
    boxes[:,3] += boxes[:,1] 

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

    class_names, num_classes = get_class_names(args1.weight)
    config1 = get_config(args1.weight, ignore_keys)
    config2 = get_config(args2.weight, ignore_keys)
    config3 = get_config(args3.weight, ignore_keys)
    config4 = get_config(args4.weight, ignore_keys)

    result_dict1 = detect(args1, config1)
    result_dict2 = detect(args2, config2)
    result_dict3 = detect(args3, config3)
    result_dict4 = detect(args4, config4)


    merged_boxes = np.array([
        np.array(result_dict1['boxes']), 
        np.array(result_dict2['boxes']), 
        np.array(result_dict3['boxes']), 
        np.array(result_dict4['boxes'])])
    merged_labels = np.array([
        np.array(result_dict1['labels']), 
        np.array(result_dict2['labels']), 
        np.array(result_dict3['labels']), 
        np.array(result_dict4['labels'])])
    merged_scores = np.array([
        np.array(result_dict1['scores']), 
        np.array(result_dict2['scores']), 
        np.array(result_dict3['scores']), 
        np.array(result_dict4['scores'])])

    merged_boxes[:,:,2] += merged_boxes[:,:,0]  #xyxy
    merged_boxes[:,:,3] += merged_boxes[:,:,1]  #xyxy

  
    final_boxes, final_scores, final_classes = box_fusion(
        merged_boxes,
        merged_scores,
        merged_labels,
        mode="wbf",
        image_size=image_size, 
        iou_threshold=0.9,
        weights = [0.25, 0.25, 0.25, 0.25]
    )

    indexes = np.where(final_scores > 0.001)[0]
    final_boxes = final_boxes[indexes]
    final_scores = final_scores[indexes]
    final_classes = final_classes[indexes]

   
    return {
        'boxes': final_boxes,
        'labels': final_classes,
        'scores': final_scores
    }, class_names

def append_food_info(food_dict, class_names):
    food_labels = food_dict['labels']
    food_names = [class_names[int(i)] for i in food_labels]
    food_info = get_info_from_db(food_names)
    food_dict.update(food_info)
    food_dict['names'] = food_names
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

    h,w,c = image.shape
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

    #xyxy box, cv2 image h,w,c
    return image[int(new_box[1]):int(new_box[3]), int(new_box[0]):int(new_box[2]), :]

    

def label_enhancement(image, cache_name, result_dict):
    boxes = np.array(result_dict['boxes'])
    labels = np.array(result_dict['labels'])
    boxes[:,2] += boxes[:,0]  #xyxy
    boxes[:,3] += boxes[:,1]  #xyxy
    
    cropped_folder = os.path.join(CACHE_DIR, cache_name)
    os.makedirs(cropped_folder, exist_ok=True)
    # Label starts at 1
    img_list = []
    new_id_list = []
 
    for box_id, (box, label) in enumerate(zip(boxes, labels)):
        if label == 21: # other food 31
            cropped = crop_box(image, box) # rgb
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

def get_prediction(
    input_path, 
    output_path,
    model_name,
    ensemble=False,
    min_iou=0.5,
    min_conf=0.1,
    enhance_labels=False):

    ignore_keys = [
            'min_iou_val',
            'min_conf_val',
            'tta',
            'gpu_devices',
            'tta_ensemble_mode',
            'tta_conf_threshold',
            'tta_iou_threshold',
        ]
    
    # get hashed key from image path
    hashed_key = os.path.basename(input_path)[:-4]

    # additional tags
    model_tag = model_name[-1]
    ensemble_tag = 'ens' if ensemble else ''

    if ensemble:
        hashed_key += f"_{ensemble_tag}"
    else:
        hashed_key += f"_{model_tag}"

    # check whether cache exists
    if check_cache(hashed_key):
        print(f"Load cache from {hashed_key}")
        class_names, _ = get_class_names(f'./{CACHE_DIR}/{model_name}.pth')
        result_dict = load_cache(hashed_key)
    else:
        if not ensemble:
            args = Arguments(model_name=model_name)
            class_names, _ = get_class_names(args.weight)

            config = get_config(args.weight, ignore_keys)
            if config is None:  
                print("Config not found. Load configs from configs/configs.yaml")
                config = Config(os.path.join('model/configs','configs.yaml'))
            else:
                print("Load configs from weight")   

            args.input_path = input_path
            result_dict = detect(args, config)
        
        else:
            img = cv2.imread(input_path)
            h,w,_ = img.shape
            result_dict, class_names = ensemble_models(input_path, [w,h]) 
        save_cache(result_dict, hashed_key)
        print(f"Save cache to {hashed_key}")
        
    class_names.insert(0, "Background")

    ori_img = cv2.imread(input_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = ori_img.shape

    # post process
    result_dict = postprocess(result_dict, img_w, img_h, min_iou, min_conf)

    # add food infomation
    result_dict = append_food_info(result_dict, class_names)

    # enhance by using a classifier
    if enhance_labels:
        result_dict = label_enhancement(ori_img, hashed_key, result_dict)

    # draw result
    draw_image(output_path, ori_img, result_dict, class_names)

    result_list = convert_dict_to_list(result_dict)
    return output_path, result_list

