import torch.nn as nn


class MaxPool2D(nn.Module):
    def __init__(self, batch_first=False):
        self.batch_first = batch_first
        super().__init__()

    def forward(self, x, mask, batch_first=False):
        x.masked_fill_(mask.logical_not().unsqueeze(-1), -1e3)
        # x.masked_fill_((~mask).unsqueeze(-1), -1e3) # for trace
        if self.batch_first:
            pool = x.max(1)[0]
        else:
            pool = x.max(0)[0]
        return pool


class MaxPoolND(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask, dim=1):
        x.masked_fill_(mask.logical_not().unsqueeze(-1), -1e3)
        # x.masked_fill_((~mask).unsqueeze(-1), -1e3) # for trace
        pool = x.max(dim)[0]
        return pool


class MeanPoolND(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask, dim=1):
        x.masked_fill_(mask.bool().logical_not().unsqueeze(-1), 0)
        # x.masked_fill_((~mask.bool()).unsqueeze(-1), 0) # for trace
        pool = x.sum(dim) / (mask.sum(dim).unsqueeze(-1) + 1e-3)
        return pool
