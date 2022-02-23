from theseus.base.models import MODEL_REGISTRY

from .segmodels import BaseSegModel
from .wrapper import ModelWithLoss

MODEL_REGISTRY.register(BaseSegModel)
