from theseus.base.models import MODEL_REGISTRY

from .yolov5 import YOLOv5
from .yolov8 import YOLOv8

MODEL_REGISTRY.register(YOLOv5)
MODEL_REGISTRY.register(YOLOv8)
