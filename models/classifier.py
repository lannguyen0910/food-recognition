import torch
from .base_model import BaseModel

class Classifier(BaseModel):
    def __init__(self, model, **kwargs):
        super(Classifier, self).__init__(**kwargs)
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
        outputs = self.model(batch, self.device)
        targets =  batch['targets'].to(self.device)
        loss = self.criterion(outputs, targets)
        loss_dict = {'T': loss.item()}
        return loss, loss_dict

    def inference_step(self, batch, return_probs=False):
        outputs = self.model(batch, self.device)
        preds = torch.argmax(outputs, dim=1)
        preds = preds.detach()
        if return_probs:
            probs = torch.nn.functional.softmax(outputs, dim=1)
            probs, _ = torch.max(probs, dim=1)
            return preds.cpu().numpy(), probs.cpu().numpy()
        else:
            return preds.numpy()

    def evaluate_step(self, batch):
        outputs = self.model(batch, self.device)
        targets =  batch['targets'].to(self.device)
        loss = self.criterion(outputs, targets)
        loss_dict = {'T': loss.item()}

        self.update_metrics(outputs = outputs, targets = targets)
        return loss, loss_dict