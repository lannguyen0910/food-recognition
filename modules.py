from model.utils.postprocess import box_fusion
import cv2
import numpy as np
import os
import pandas as pd
from model import detect, get_config, Config, download_weights, draw_boxes_v2, get_class_names, postprocessing

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
            tmp_path = os.path.join(CACHE_DIR, self.model_name)
            download_pretrained_weights(
                self.model_name, 
                cached=tmp_path)
            self.weight=tmp_path
            
weight_urls = {
    'yolov5m': "1-EDbsoPOlYlkZGjol5sDSG4bhlJbgkDI"
}

def download_pretrained_weights(name, cached=None):
    return download_weights(weight_urls[name], cached)


def draw_image(out_path, ori_img, result_dict, class_names):
    draw_boxes_v2(
        out_path, 
        ori_img , 
        result_dict["boxes"], 
        result_dict["labels"], 
        result_dict["scores"], 
        class_names)


def cache_prediction(result_dict, cache_name):
    df = pd.DataFrame(result_dict)
    df.to_csv(f'./{CACHE_DIR}/{cache_name}.csv', index=False)

def load_cache(image_name):
    df = pd.read_csv(f'./{CACHE_DIR}/{image_name}.csv')
    result_dict = {
        'boxes': [],
        'labels': [],
        'scores': []
    }
    for idx, row in df.iterrows():
        box, label, score = row
        result_dict['boxes'].append(box)
        result_dict['labels'].append(label)
        result_dict['scores'].append(score)

    return result_dict

def postprocess(args, result_dict, img_w, img_h):
    outputs = {
        'bboxes': result_dict['boxes'],
        'scores': result_dict['scores'],
        'classes': result_dict['labels']
    }

    outputs = postprocessing(
        outputs, 
        current_img_size=[img_w, img_h],
        min_iou=args.min_iou,
        min_conf=args.min_conf,
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

    final_boxes, final_scores, final_classes = box_fusion(
        [result_dict1['boxes'], result_dict2['boxes'], result_dict3['boxes'], result_dict4['boxes']],
        [result_dict1['labels'] + result_dict2['labels'] + result_dict3['labels'] + result_dict4['labels']],
        [result_dict1['scores'] + result_dict2['scores'] + result_dict3['scores'] + result_dict4['scores']],
        mode="wbf",
        image_size=image_size, 
        iou_threshold=0.9,
        weights = [0.25, 0.25, 0.25, 0.25]
    )

    indexes = np.where(final_scores > 0.01)[0]
    final_boxes = final_boxes[indexes]
    final_scores = final_scores[indexes]
    final_classes = final_classes[indexes]

    return {
        'boxes': final_boxes,
        'labels': final_classes,
        'scores': final_scores
    }

def get_prediction(
    input_path, 
    output_path,
    model_name,
    ensemble=False):

    ignore_keys = [
            'min_iou_val',
            'min_conf_val',
            'tta',
            'gpu_devices',
            'tta_ensemble_mode',
            'tta_conf_threshold',
            'tta_iou_threshold',
        ]
    
    if not ensemble:
        args = Arguments(model_name=model_name)
        class_names, num_classes = get_class_names(args.weight)

        config = get_config(args.weight, ignore_keys)
        if config is None:  
            print("Config not found. Load configs from configs/configs.yaml")
            config = Config(os.path.join('model/configs','configs.yaml'))
        else:
            print("Load configs from weight")   

        args.input_path = input_path
        result_dict = detect(args, config)
      
    else:
        result_dict = ensemble_models(input_path)

    cache_prediction(result_dict)
    ori_img = cv2.imread(input_path)
    draw_image(output_path, ori_img, result_dict, class_names)

    return output_path, result_dict

