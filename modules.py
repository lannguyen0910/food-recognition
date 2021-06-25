import os
from model import detect, get_config, Config, download_weights

CACHE_DIR = '.cache'
WEIGHT = "./yolov5m_best.pth"

class Arguments:
    def __init__(self, model_name='yolov5m') -> None:
        self.model_name = model_name
        self.weight = None
        self.input_path=""
        self.output_path=""
        self.gpus="0"
        self.min_conf=0.1
        self.min_iou=0.5
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


def get_prediction(
    input_path, 
    output_path,
    model_name):

    ignore_keys = [
            'min_iou_val',
            'min_conf_val',
            'tta',
            'gpu_devices',
            'tta_ensemble_mode',
            'tta_conf_threshold',
            'tta_iou_threshold',
        ]


    args = Arguments(
        model_name=model_name
    )


    config = get_config(args.weight, ignore_keys)
    if config is None:  
        print("Config not found. Load configs from configs/configs.yaml")
        config = Config(os.path.join('model/configs','configs.yaml'))
    else:
        print("Load configs from weight")   

    args.input_path = input_path
    args.output_path = output_path
    result_dict = detect(args, config)

    return output_path, result_dict

