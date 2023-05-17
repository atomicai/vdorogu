import ctypes
import itertools
import logging
import os
import sys
from collections import Counter, defaultdict
from ctypes import cdll

import numpy as np
import torch
from torch.utils.data import IterableDataset

so_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../bin/libqltreader.so'))
# print(so_path)
libqltreader = cdll.LoadLibrary(so_path)
libqltreader.QLTReader_new.restype = ctypes.c_void_p
libqltreader.QLTReader_new.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
libqltreader.QLTReader_delete.argtypes = [ctypes.c_void_p]
libqltreader.QLTReader_Remaining.restype = ctypes.c_bool
libqltreader.QLTReader_Remaining.argtypes = [ctypes.c_void_p]
libqltreader.QLTReader_Print.argtypes = [ctypes.c_void_p]
libqltreader.QLTReader_Write.restype = ctypes.c_int
libqltreader.QLTReader_Write.argtypes = [ctypes.c_void_p] * 5


class Recover:
    def __init__(self, dict_path):
        self.words = []
        with open(dict_path) as f:
            for word in f:
                word, cnt = word.strip().split('\t')
                word = word.strip().lower()
                self.words.append(word.strip())

    def __call__(self, idxs):
        return self.recover(idxs)

    def recover(self, idxs):
        return ' '.join(self.words[i - 2] if i != 1 else '<UNK>' for i in idxs if i != 0)


class Indexer:
    def __init__(self, dict_path):
        self.word2idx = {}
        idx = 1
        with open(dict_path) as f:
            for word in f:
                word, cnt = word.strip().split('\t')
                word = word.strip().lower()
                self.word2idx[word] = idx
                idx += 1

    def __call__(self, words):
        return self.index(words)

    def index(self, words):
        return [self.word2idx.get(w.strip().lower(), 0) + 1 for w in words]


class QLTReader(object):
    def __init__(self, qlt_path, batch_size, num_queries):
        self.queries = np.zeros([num_queries, 20], dtype=np.int32)
        self.group_sizes = np.zeros([num_queries], dtype=np.int32)
        self.titles = np.zeros([batch_size, 20], dtype=np.int32)
        self.labels = np.zeros([batch_size], dtype=np.int32)
        self.obj = libqltreader.QLTReader_new(qlt_path.encode('UTF-8'), batch_size, num_queries)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        libqltreader.QLTReader_delete(self.obj)

    def __iter__(self):
        while self.remaining():
            self.read()
            nq = self.num_queries
            num_pairs = np.sum(self.group_sizes)
            yield self.queries[:nq], self.group_sizes[:nq], self.titles[:num_pairs], self.labels[:num_pairs]

    def remaining(self):
        return libqltreader.QLTReader_Remaining(self.obj)

    def print(self):
        libqltreader.QLTReader_Print(self.obj)

    def read(self):
        queries = self.queries.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        group_sizes = self.group_sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        titles = self.titles.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        labels = self.labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.num_queries = libqltreader.QLTReader_Write(self.obj, queries, group_sizes, titles, labels)
        return self.remaining()


class MultifileQLT(IterableDataset):
    def __init__(self, path, num_pairs=3072, num_queries=384):
        super(MultifileQLT).__init__()
        self.files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.bin')]
        self.num_pairs = num_pairs
        self.num_queries = num_queries
        logging.warning('Found %d binary objects' % len(self.files))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        for fname in self.files:
            with QLTReader(fname, self.num_pairs, self.num_queries) as r:
                if worker_info is None:
                    for batch in r:
                        yield batch
                else:
                    worker_num = worker_info.num_workers
                    worker_id = worker_info.id
                    for batch_idx, batch in enumerate(r):
                        if batch_idx % worker_num == worker_id:
                            yield batch
