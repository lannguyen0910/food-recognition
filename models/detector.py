from .base_model import BaseModel
import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm


import sys
sys.path.append('..')

class Detector(BaseModel):
    def __init__(self, model, **kwargs):
        super(Detector, self).__init__(**kwargs)
        self.model = model
        self.model_name = self.model.name
        if self.optimizer is not None:
            self.optimizer = self.optimizer(self.parameters(), lr= self.lr)
            self.set_optimizer_params()

        if self.freeze:
            for params in self.model.parameters():
                params.requires_grad = False

        if self.device:
            self.model.to(self.device)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        output = self.model(batch, self.device)
        loss_dict = {k:v.item() for k,v in output.items()}
        loss = output['T']
        return loss, loss_dict

    def inference_step(self, batch):
        outputs = self.model.detect(batch, self.device) 
        return outputs  

    def evaluate_step(self, batch):
        output = self.model(batch, self.device)
        loss_dict = {k:v.item() for k,v in output.items()}
        loss = output['T']
        self.update_metrics(model=self)
        return loss, loss_dict
