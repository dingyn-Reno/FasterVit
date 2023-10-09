import torch
import torch.nn as nn
from model import net

model=net.Model()
model_dict=model.state_dict()
weight=torch.load('runs-1-3750.ppt')

pretrained_dict = {k: v for k, v in weight.items() if k in model_dict}

model_dict.update(pretrained_dict)

model.load_state_dict(model_dict)