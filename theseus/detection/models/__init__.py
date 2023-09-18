from theseus.base.models import MODEL_REGISTRY

from .yolov5 import *

MODEL_REGISTRY.register(YoloBackbone)
