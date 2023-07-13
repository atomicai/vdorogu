import torch


class BinaryRankingLoss:
    def __init__(self):
        pass

    def __call__(self, positives, negatives):
        pshape = positives.shape
        nshape = negatives.shape
        positives = positives.view(pshape[0], 1, pshape[1])
        negatives = negatives.view(nshape[0], nshape[1], 1)
        diff = positives - negatives
        diff = torch.clamp_max(-diff, 30.0)
        diff = torch.exp(diff)
        loss = torch.mean(torch.log1p(diff))
        return loss
