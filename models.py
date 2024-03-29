"""
Residual DnCNN
"""


import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=10):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.RReLU(inplace=True))
        for _ in range(num_of_layers-2):
            """
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
            """
            layers.append(residualBlock(features))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out

class residualBlock(nn.Module):
    def __init__(self,features):
        super(residualBlock,self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.BatchNorm2d(features))
        layers.append(nn.RReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.BatchNorm2d(features))
        layers.append(nn.RReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        self.residualBlock = nn.Sequential(*layers)
        
    def forward(self,x):
        out = self.residualBlock(x)
        return out + x
    
        