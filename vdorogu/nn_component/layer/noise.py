import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach

    def forward(self, x):
        if self.training and self.sigma != 0:  
            device = x.device
            dtype = x.dtype
            noise = torch.randn(x.size()).type(dtype)
            noise = noise.to(device)
            return x + Variable(noise * self.sigma)            
        return x
