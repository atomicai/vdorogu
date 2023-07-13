import numpy as np


class SparseContainer:
    def __init__(self, data):
        self.data = np.concatenate(data).astype(np.int32)
        self.indptr = np.cumsum([0] + [len(x) for x in data]).astype(np.int64)

    def __getitem__(self, idx):
        if isinstance(idx, np.ndarray) or isinstance(idx, list):
            data = [self[x] for x in idx]
            return self.__class__(data)
        start = self.indptr[idx]
        end = self.indptr[idx + 1]
        return self.data[start:end]

    def __len__(self):
        return self.indptr.shape[0] - 1

    @property
    def shape(self):
        return (self.indptr.shape[0] - 1,)


class UniqueSparseContainer:
    def __init__(self, container, qids):
        _, query_index = np.unique(qids, return_index=True)
        self.unique_sparse_container = container[query_index]
        self.mapper = qids

    def __getitem__(self, idx):
        return self.unique_sparse_container[self.mapper[idx]]

    def __len__(self):
        return len(self.mapper)

    @property
    def shape(self):
        return (self.mapper,)
