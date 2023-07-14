import math
from itertools import groupby

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_EPSILON = 1e-10


class PairwiseLogisticLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pairwise_logits):
        # The following is the same as log(1 + exp(-pairwise_logits)).
        return torch.relu(-pairwise_logits) + torch.log1p(torch.exp(-torch.abs(pairwise_logits)))


class MarginMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, student_scores, teacher_scores):
        if len(student_scores.shape) == 1:
            student_scores = student_scores.view(-1, 1)

        if len(teacher_scores.shape) == 1:
            teacher_scores = teacher_scores.view(-1, 1)

        teacher_diff = (teacher_scores - teacher_scores.T).view(1, -1)
        student_diff = (student_scores - student_scores.T).view(1, -1)

        teacher_diff = F.normalize(teacher_diff, p=2, dim=-1)
        student_diff = F.normalize(student_diff, p=2, dim=-1)

        teacher_diff *= len(teacher_diff) * 1e7
        student_diff *= len(student_diff) * 1e7

        return self.mse(teacher_diff, student_diff)


class PairwiseHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pairwise_logits):
        return torch.relu(1 - pairwise_logits)


class PairwiseSoftZeroOneLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pairwise_logits):
        return torch.where(pairwise_logits > 0, 1.0 - torch.sigmoid(pairwise_logits), torch.sigmoid(-pairwise_logits))


def power_gain(x):
    return 2**x - 1


def identity_gain(x):
    return x


def log_gain(x):
    return torch.log1p(x)


def log_power_gain(x):
    return 2 ** torch.log1p(x) - 1


class RankingLossListWise(nn.Module):
    def __init__(self, pairwise_loss=PairwiseLogisticLoss(), topn=50, smooth_fraction=0.0, gain_fn=power_gain):
        super().__init__()
        self.topn = topn
        self.pairwise_loss = pairwise_loss
        self.smooth_fraction = smooth_fraction
        self.gain_fn = gain_fn

    def forward(self, logits, labels):
        is_valid = labels > -1000
        labels = labels.float()
        labels = torch.where(is_valid, labels, torch.zeros_like(labels))
        logits = torch.where(is_valid, logits, math.log(_EPSILON) * torch.ones_like(logits))
        # pairwise diffs and masks
        pairwise_logits_diff = logits.unsqueeze(-1) - logits.unsqueeze(-2)
        with torch.no_grad():
            pairwise_labels_diff = labels.unsqueeze(-1) - labels.unsqueeze(-2)
            valid_mask = (is_valid.unsqueeze(-1) * is_valid.unsqueeze(-2)).float()
            labels_mask = (pairwise_labels_diff > 0).float()
            mask = labels_mask * valid_mask

            # pairwise discounts
            ranks = (torch.argsort(torch.argsort(logits, -1, descending=True), -1) + 1).float()
            discount = torch.where(ranks > self.topn, torch.zeros_like(ranks), 1.0 / torch.log1p(ranks))
            absoulte_discouts = torch.abs(discount.unsqueeze(-1) - discount.unsqueeze(-2))

            capped_rank = torch.where(ranks > self.topn, torch.ones_like(ranks) * (self.topn + 1), ranks)
            rank_diff = torch.abs(capped_rank.unsqueeze(-1) - capped_rank.unsqueeze(-2))
            relative_discouts = torch.where(
                rank_diff > 0,
                torch.abs(1.0 / torch.log1p(rank_diff) - 1.0 / torch.log1p(rank_diff + 1)),
                torch.zeros_like(rank_diff),
            )

            pairwise_discouts = (1 - self.smooth_fraction) * relative_discouts + self.smooth_fraction * absoulte_discouts

            # max_dcg
            sorted_labels = torch.gather(labels, -1, torch.argsort(labels, -1, descending=True))
            ranks = (torch.arange(labels.shape[-1], dtype=labels.dtype, device=labels.device).view(1, -1) + 1).float()
            discounted_gain = self.gain_fn(sorted_labels) * (1.0 / torch.log1p(ranks))
            discounted_gain = torch.where(ranks > self.topn, torch.zeros_like(labels), discounted_gain)
            max_dcg = torch.sum(discounted_gain, -1, keepdim=True)
            max_idcg = torch.where(max_dcg > 0, 1 / max_dcg, torch.zeros_like(max_dcg))

            # pairwise gains
            gain = self.gain_fn(labels)
            gain *= max_idcg
            pairwise_gains = torch.abs(gain.unsqueeze(-1) - gain.unsqueeze(-2))
            weights = pairwise_discouts * pairwise_gains
        loss = self.pairwise_loss(pairwise_logits_diff) * weights * mask
        loss = loss.sum(-1).sum(-1)
        return loss.mean()


class RankingLossListWiseDistil(nn.Module):
    def __init__(self, topn=50, smooth_fraction=0.25):
        super().__init__()
        self.topn = topn
        self.pairwise_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.smooth_fraction = smooth_fraction
        self.temperature = 2.0

    def forward(self, logits, labels):
        is_valid = labels > -1000
        labels = labels.float()
        labels = torch.where(is_valid, labels, torch.zeros_like(labels))
        logits = torch.where(is_valid, logits, math.log(_EPSILON) * torch.ones_like(logits))
        # pairwise diffs and masks
        pairwise_logits_diff = logits.unsqueeze(-1) - logits.unsqueeze(-2)
        with torch.no_grad():
            pairwise_labels_diff = labels.unsqueeze(-1) - labels.unsqueeze(-2)
            valid_mask = (is_valid.unsqueeze(-1) * is_valid.unsqueeze(-2)).float()
            labels_mask = (pairwise_labels_diff > 0).float()
            mask = labels_mask * valid_mask

            # pairwise discounts
            ranks = (torch.argsort(torch.argsort(logits, -1, descending=True), -1) + 1).float()
            discount = torch.where(ranks > self.topn, torch.zeros_like(ranks), 1.0 / torch.log1p(ranks))
            absoulte_discouts = torch.abs(discount.unsqueeze(-1) - discount.unsqueeze(-2))

            capped_rank = torch.where(ranks > self.topn, torch.ones_like(ranks) * (self.topn + 1), ranks)
            rank_diff = torch.abs(capped_rank.unsqueeze(-1) - capped_rank.unsqueeze(-2))
            relative_discouts = torch.where(
                rank_diff > 0,
                torch.abs(1.0 / torch.log1p(rank_diff) - 1.0 / torch.log1p(rank_diff + 1)),
                torch.zeros_like(rank_diff),
            )

            pairwise_discouts = (1 - self.smooth_fraction) * relative_discouts + self.smooth_fraction * absoulte_discouts

            # max_dcg
            ranks = (torch.arange(labels.shape[-1], dtype=labels.dtype, device=labels.device).view(1, -1) + 1).float()
            discounted_gain = (1.0 / torch.log1p(ranks)) * is_valid.float()
            discounted_gain = torch.where(ranks > self.topn, torch.zeros_like(labels), discounted_gain)
            max_dcg = torch.sum(discounted_gain, -1, keepdim=True)
            max_idcg = torch.where(max_dcg > 0, 1 / max_dcg, torch.zeros_like(max_dcg))

            weights = pairwise_discouts  # * pairwise_gains

        loss = self.pairwise_loss(
            (pairwise_logits_diff / self.temperature), (pairwise_labels_diff / self.temperature).sigmoid()
        )
        loss = loss * weights * mask
        loss = loss.sum(-1) * max_idcg
        loss = loss.sum(-1)
        return loss.mean()
