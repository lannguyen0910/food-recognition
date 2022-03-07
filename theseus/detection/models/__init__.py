from theseus.base.models import MODEL_REGISTRY

from .yolo import *

MODEL_REGISTRY.register(YoloBackbone)
