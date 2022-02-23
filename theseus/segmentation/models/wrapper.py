import torch
from torch import nn

class ModelWithLoss(nn.Module):
    """Add utilitarian functions for module to work with pipeline

    Args:
        model (Module): Base Model without loss
        loss (Module): Base loss function with stat

    """

    def __init__(self, model: nn.Module, criterion: nn.Module, device: torch.device):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.device = device

    def forward(self, batch, metrics=None):
        outputs = self.model(batch["inputs"].to(self.device))
        loss, loss_dict = self.criterion(outputs, batch, self.device)

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