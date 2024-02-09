import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(conv_block).__init__(*args, **kwargs)
        self.relu=nn.ReLU()
        self.conv=nn.Conv2d(in_channels,out_channels,**kwargs)
        self.batchnorm=nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return self.relu(self.batchnorm(self.conv(x)))

