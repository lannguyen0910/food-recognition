from torch import nn

from . import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ModelWithLoss(nn.Module):
    """Wrapper for model with loss function

    Args:
        model (Module): Base Model without loss
        loss (Module): Base loss function with stat

    """

    def __init__(self, model: nn.Module, criterion: nn.Module):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, batch, metrics=None):
        outputs = self.model(batch["inputs"])
        loss, loss_dict = self.criterion(outputs, batch)

        if metrics is not None:
            for metric in metrics:
                metric.update(outputs, batch)

        return {
            'loss': loss,
            'loss_dict': loss_dict
        }

    def training_step(self, batch):
        return self.forward(batch)

    def evaluate_step(self, batch, metrics=None):
        return self.forward(batch, metrics)

    def state_dict(self):
        return self.model.state_dict()

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)