import torch
import torch.nn as nn
import torch.nn.functional as F

# from apex.contrib.xentropy.softmax_xentropy import SoftmaxCrossEntropyLoss

from torch.nn import CrossEntropyLoss

"""
class CrossEntropyLoss(nn.Module):
    def forward(self, logits, labels):
        return SoftmaxCrossEntropyLoss.apply(logits, labels,0,1,True).mean()
"""


class DistilCELoss(nn.Module):
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, s_logits, t_logits):
        loss_ce = -torch.sum(
            torch.log_softmax(s_logits / self.temperature, -1) * torch.softmax(t_logits / self.temperature, -1),
            -1,
            dtype=torch.float,
        ).mean()
        return loss_ce * (self.temperature**2)


class DistilKLLoss(nn.Module):
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, s_logits, t_logits):
        p_student = torch.softmax(s_logits / self.temperature, -1)
        p_teacher = torch.softmax(t_logits / self.temperature, -1)

        loss_kl = torch.sum(p_teacher * (torch.log(p_teacher) - torch.log(p_student)), -1, dtype=torch.float).mean()
        return loss_kl * (self.temperature**2)
