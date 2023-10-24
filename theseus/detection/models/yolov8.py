import numpy as np

import torch
from ultralytics import YOLO
from .backbone import BaseBackbone


class YOLOv8(BaseBackbone):
    """
    Some yolov8 models with various pretrained backbones from hub

    weight : `str` 
        weight path to load custom yolov5 weight
    min_conf: `float` 
        NMS confidence threshold
    min_iou: `float`
        NMS IoU threshold
    max_det: `int` 
        maximum number of detections per image - 300 for YOLO
    """

    def __init__(
            self,
            weight: str,
            min_iou: float,
            min_conf: float,
            max_det: int = 300,
            **kwargs):

        super().__init__(**kwargs)
        self.model = YOLO(weight)

        self.class_names = self.model.names

        self.conf = min_conf  # NMS confidence threshold
        self.iou = min_iou  # NMS IoU threshold
        self.max_det = max_det  # maximum number of detections per image

    def get_model(self):
        """
        Return the full architecture of the model, for visualization
        """
        return self.model

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        return outputs

    def get_prediction(self, image: str):
        out = []
        results = self.model.predict(image, conf=self.conf, iou=self.iou, max_det=self.max_det)  # inference

        for result in results:
            bboxes = []
            labels = []
            scores = []

            boxes = result.boxes.cpu().numpy()
            for _, box in enumerate(boxes):
                r = box.xyxy[0].astype(int)
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                bboxes.append(r)
                labels.append(cls)
                scores.append(conf)
            
            if len(bboxes) == 0:
                continue

            if len(bboxes) > 0:
                out.append({
                    'bboxes': np.array(bboxes),
                    'classes': np.array(labels),
                    'scores': np.array(scores),
                })

        return out