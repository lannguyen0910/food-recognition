from theseus.base.models import MODEL_REGISTRY

from .timm_models import *
from .multiheads import *
from .wrapper import ModelWithLoss

MODEL_REGISTRY.register(BaseTimmModel)
MODEL_REGISTRY.register(MultiHeadModel)