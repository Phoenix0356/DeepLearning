import torchvision

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import Util
import os
import numpy as np

class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()

    def forward(self,x):
        return x
model=torchvision.models.vgg16()

for param in model.parameters():
    param.requires_grad=False

model.avgpool=Identity()
model.classifier=nn.Sequential(nn.Linear(512,100),
                               nn.ReLU(),
                               nn.Linear(100,10))
print(model)