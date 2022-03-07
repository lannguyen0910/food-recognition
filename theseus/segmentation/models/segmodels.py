from typing import Dict, Any
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

"""
Source: https://github.com/qubvel/segmentation_models.pytorch
"""

class BaseSegModel(nn.Module):
    """
    Some simple segmentation models with various pretrained backbones

    name: `str`
        model name [unet, deeplabv3, ...]
    encoder_name : `str` 
        backbone name [efficientnet, resnet, ...]
    num_classes: `int` 
        number of classes
    aux_params: `Dict` 
        auxilliary head
    """
    def __init__(
        self, 
        name: str, 
        encoder_name : str = "resnet34", 
        num_classes: int = 1000,
        aux_params: Dict = None,
        **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.model = smp.create_model(
            arch = name,
            encoder_name = encoder_name,
            in_channels = 3,
            encoder_weights = "imagenet",
            classes = num_classes, 
            aux_params = aux_params)

    def get_model(self):
        """
        Return the full architecture of the model, for visualization
        """
        return self.model

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        return outputs

    def get_prediction(self, adict: Dict[str, Any], device: torch.device):
        """
        Inference using the model.

        adict: `Dict[str, Any]`
            dictionary of inputs
        device: `torch.device`
            current device 
        """
        inputs = adict['inputs'].to(device)
        outputs = self.model(inputs)

        if self.num_classes == 1:
            thresh = adict['thresh']
            predicts = (outputs > thresh).float()
        else:
            predicts = torch.argmax(outputs, dim=1)

        predicts = predicts.detach().cpu().squeeze().numpy()
        return {
            'masks': predicts
        } 

        