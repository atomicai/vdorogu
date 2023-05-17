import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn


def view_iterator(view, data):
    if isinstance(view, int):
        for i in range(0, len(data), view):
            yield data[i : i + view]
        return
    offset = 0
    for size in view:
        yield data[offset : offset + size]
        offset += size


def subsample_groups(group_sizes, titles, labels, target_size):
    dim = titles.shape[1]
    subtitles = np.zeros([len(group_sizes) * target_size, dim], dtype=titles.dtype)
    sublabels = np.zeros([len(group_sizes) * target_size], dtype=labels.dtype)
    for i, (t, l) in enumerate(zip(view_iterator(group_sizes, titles), view_iterator(group_sizes, labels))):
        choice = np.sort(np.random.choice(len(t), target_size, len(t) < target_size))
        subtitles[target_size * i : target_size * (i + 1)] = t[choice]
        sublabels[target_size * i : target_size * (i + 1)] = l[choice]
    return subtitles, sublabels


class EuclideanMatcher(nn.Module):
    def __init__(self):
        super(EuclideanMatcher, self).__init__()

    def forward(self, q, t):
        return -torch.sum((q - t) ** 2, dim=-1)


class NpCyclicBuffer:
    def __init__(self, shape, dtype):
        self.data = np.zeros(shape, dtype)
        self.reset()

    def reset(self):
        self.size = 0  # data[0:size] are valid points
        self.position = 0  # data[position] will be updated on the next push() call

    def is_full(self):
        return self.size == self.data.shape[0]

    def push(self, item):
        if not self.is_full():
            assert self.size == self.position
            self.size += 1
        self.data[self.position] = item
        self.position = (self.position + 1) % self.data.shape[0]

    def push_many(self, items):
        for item in items:
            self.push(item)

    def get_data(self):
        return self.data[: self.size]


def predict(predictor, X, batch_size, device, output_shape=None):
    output_shape = [len(X), *output_shape] if output_shape is not None else len(X)
    preds = np.empty(output_shape, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, len(preds), batch_size):
            minibatch = torch.tensor(X[i : i + batch_size]).to(device)
            pred = predictor(minibatch).cpu().data.numpy()
            preds[i : i + batch_size] = pred
    return preds
