from torch import nn


class BaseBackbone(nn.Module):
    def __init__(self, **kwargs):
        super(BaseBackbone, self).__init__()
        pass

    def forward(self, batch):
        pass

    def detect(self, batch):
        pass
