import numpy as np
import torch
import random

SEED = 1702
# Inherited from https://github.com/vltanh/pytorch-template
def seed_everything(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)