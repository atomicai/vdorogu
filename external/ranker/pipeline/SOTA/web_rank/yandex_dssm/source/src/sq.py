import numpy as np
from pipeline.SOTA.web_rank.yandex_dssm.source.src.utils import view_iterator
from torch.utils.data import Dataset


def calc_sq_dcg(preds, labels, at):
    order = np.argsort(-preds)[:at]
    return np.sum((2 ** labels[order] - 1) / np.log2(2 + np.arange(len(order))))


class SQDataset(Dataset):
    def __init__(self, path, indexer, max_token_len):
        super(SQDataset).__init__()
        queries, labels, titles, qids = [], [], [], []
        with open(path) as f:
            for line in f:
                parts = line.strip('\n').split('\t')
                parts.extend([''] * (4 - len(parts)))
                q, t, l, qid = parts
                queries.append(q)
                labels.append(int(l))
                titles.append(t)
                qids.append(int(qid))
        self.queries_raw = queries
        self.titles_raw = titles
        self.labels = np.array(labels)
        self.qids = np.array(qids)
        self.queries = np.zeros([len(labels), max_token_len], dtype=np.int32)
        self.titles = np.zeros([len(labels), max_token_len], dtype=np.int32)
        self.sizes = []
        cur_size = 0
        cur_q = queries[0]
        for i, (q, t) in enumerate(zip(queries, titles)):
            if cur_q != q:
                cur_q = q
                self.sizes.append(cur_size)
                cur_size = 0
            cur_size += 1
            q = indexer(q.split())[:max_token_len]
            t = indexer(t.split())[:max_token_len]
            self.queries[i, : len(q)] = q
            self.titles[i, : len(t)] = t
        self.sizes.append(cur_size)
        self.sizes = np.array(self.sizes, dtype=np.int32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.queries[idx], self.titles[idx], self.labels[idx], self.qids[idx]

    def ndcg(self, predictions, at=5):
        ndcgs = []
        for p, l in zip(view_iterator(self.sizes, predictions), view_iterator(self.sizes, self.labels)):
            dcg = calc_sq_dcg(p, l, at)
            norm = calc_sq_dcg(l, l, at) + 1e-5
            ndcgs.append(dcg / norm)
        return np.mean(ndcgs)
