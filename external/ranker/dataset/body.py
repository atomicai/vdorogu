import gc
import gzip
import json
import os
import pickle
import random
import unicodedata
from itertools import chain

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler

from external.ranker.dataset.web import bert_concat, t5_concat
from external.ranker.nn_component.model.nlp.parade import get_indexer


def preprocess(text):
    text = " ".join(text.strip().split())
    text = text.replace("``", '"').replace("''", '"')
    nfkd_form = unicodedata.normalize("NFKD", text)
    text = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return text.lower()


def encode(tokenizer, inp):
    inp = preprocess(inp)
    inp = tokenizer.encode(inp)  # ,dropout_prob=0.1)
    return np.array(inp).astype(np.int64)


class BodyDataset(IterableDataset):
    def __init__(self, train_dataset, tokenizer, max_query, query_maxlen, body_maxlen, mapper=None, dssm_variant=True):
        super().__init__()

        files = sorted([os.path.join(train_dataset, p) for p in os.listdir(train_dataset) if p.endswith(".gz")])
        self.fnames = np.array(list(files))

        self.tokenizer = tokenizer
        self.mapper = mapper
        self.rank = None

        self.max_query = max_query
        self.query_maxlen = query_maxlen
        self.body_maxlen = body_maxlen

        self.dummy_token_ids = np.array([tokenizer.bos_token_id, tokenizer.eos_token_id])
        self.dummy_label = -1000.0

        self.dssm_variant = dssm_variant

        self.custom_processor = None

    def set_rank(self, rank, ranks_num):
        self.rank = rank
        self.ranks_num = ranks_num

        if len(self.fnames) < ranks_num:
            return [self.fnames[rank % len(self.fnames)]]

        chunk_size = (len(self.fnames) + ranks_num - 1) // ranks_num
        start = rank * chunk_size
        end = (rank + 1) * chunk_size
        self.fnames = self.fnames[start:end]

        print(rank)

        return self.fnames

    def prepare_ids(self, ids, max_len):
        result = ids
        if self.mapper:
            # print(max(content))
            result = [int(self.mapper.get(token, 3)) for token in ids]
            # print(max(content))

        result = np.array(result[: max_len - 2])
        if self.dssm_variant:
            result = np.concatenate([[self.tokenizer.bos_token_id], result, [self.tokenizer.eos_token_id]])

        return result

    @property
    def len(self):
        return 670420

    def __len__(self):
        return self.len

    def __iter__(self):
        assert self.rank is not None

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        if len(self.fnames) < self.ranks_num:
            worker_id = worker_id + num_workers * (self.rank // len(self.fnames))
            num_workers *= (self.ranks_num + len(self.fnames) - 1) // len(self.fnames)

            print(self.rank, worker_id, num_workers)

        iterations = 0

        while True:
            # np.random.shuffle(self.fnames)
            # print(self.rank, worker_id, self.fnames)
            for fname in self.fnames:
                # print(worker_id, fname)
                with gzip.open(fname) as f:
                    for i, line in enumerate(f):
                        if i % num_workers != worker_id:
                            continue
                        try:
                            line = line.decode().strip()
                            data = json.loads(line)
                        except:
                            continue

                        batch = []

                        query_unprocessed = data['query'][0]
                        query = self.prepare_ids(query_unprocessed, self.query_maxlen)
                        for data in data['data']:
                            label = data['xlm_roberta_large_query64_body440']

                            if self.custom_processor is not None:
                                query, document = self.custom_processor(self.tokenizer, query_unprocessed, data['body'])
                            else:
                                document = self.prepare_ids(data['body'], self.body_maxlen)

                            if self.dssm_variant:
                                batch.append((query, document, label))
                            else:
                                # token_types = np.concatenate([[0], np.zeros_like(query) + 1, [0, 0], np.zeros_like(document) + 2, [0]])
                                result = np.concatenate(
                                    [
                                        [self.tokenizer.bos_token_id],
                                        query,
                                        [self.tokenizer.sep_token_id, self.tokenizer.sep_token_id],
                                        document,
                                        [self.tokenizer.eos_token_id],
                                    ]
                                )
                                batch.append((result, label))

                        while len(batch) < self.max_query:
                            if self.dssm_variant:
                                batch.append((query, self.dummy_token_ids, self.dummy_label))
                            else:
                                # token_types = np.concatenate([[0], np.zeros_like(query) + 1, [0, 0], np.zeros_like(self.dummy_token_ids) + 2, [0]])
                                result = np.concatenate(
                                    [
                                        [self.tokenizer.bos_token_id],
                                        query,
                                        [self.tokenizer.sep_token_id, self.tokenizer.sep_token_id],
                                        self.dummy_token_ids,
                                        [self.tokenizer.eos_token_id],
                                    ]
                                )
                                batch.append((result, self.dummy_label))

                        random.shuffle(batch)
                        batch = batch[: self.max_query]
                        yield tuple(map(list, zip(*batch)))

                        iterations += 1
                        if iterations >= self.len:
                            break
                            # pass

            break


class FlatDatasetDSSM(Dataset):
    def __init__(
        self, queries, documents, labels, qids, tokenizer, query_maxlen, body_maxlen, mapper=None, dssm_variant=True
    ):
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.qids = qids
        self.tokenizer = tokenizer
        self.mapper = mapper

        self.query_maxlen = query_maxlen
        self.body_maxlen = body_maxlen

        self.dssm_variant = dssm_variant

        self.custom_processor = None

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx][: self.query_maxlen - 2]
        document = self.documents[idx][: self.body_maxlen - 2]
        if self.mapper:
            query = np.array([self.mapper.get(i, 3) for i in query], dtype=np.int32)
            document = np.array([self.mapper.get(i, 3) for i in document], dtype=np.int32)

        if self.dssm_variant:
            if self.custom_processor is not None:
                query, document = self.custom_processor(self.tokenizer, query, document)
            else:
                query = np.concatenate([[self.tokenizer.bos_token_id], query, [self.tokenizer.eos_token_id]])
                document = np.concatenate([[self.tokenizer.bos_token_id], document, [self.tokenizer.eos_token_id]])

            return query, document, self.labels[idx], self.qids[idx]
        else:
            inputs = np.concatenate(
                [
                    [self.tokenizer.bos_token_id],
                    query,
                    [self.tokenizer.sep_token_id, self.tokenizer.sep_token_id],
                    document,
                    [self.tokenizer.eos_token_id],
                ]
            )
            return inputs, self.labels[idx], self.qids[idx]


from dataset.web import get_indptr


class QueryDatasetDSSM(Dataset):
    def __init__(
        self,
        queries,
        documents,
        labels,
        qids,
        tokenizer,
        max_query,
        query_maxlen,
        body_maxlen,
        mapper=None,
        priority_label=None,
    ):
        '''
        priority_label - the label that must be sampled
        '''
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.indptr = get_indptr(qids)
        self.tokenizer = tokenizer
        self.max_query = max_query
        self.query_maxlen = query_maxlen
        self.body_maxlen = body_maxlen
        self.dummy_token_id = np.array([tokenizer.bos_token_id, tokenizer.eos_token_id])
        self.dummy_label = -1000.0
        self.mapper = mapper
        self.priority_label = priority_label

    def __len__(self):
        return len(self.indptr) - 1

    def __getitem__(self, idx):
        begidx = self.indptr[idx]
        endidx = self.indptr[idx + 1]
        indices = np.arange(begidx, endidx)
        query_ids = []
        document_ids = []
        labels = []

        for query, document, label in zip(self.queries[indices], self.documents[indices], self.labels[indices]):
            if self.mapper:
                query = np.array([self.mapper.get(i, 3) for i in query], dtype=np.int32)
                document = np.array([self.mapper.get(i, 3) for i in document], dtype=np.int32)

            query_ids.append(
                np.concatenate(
                    [[self.tokenizer.bos_token_id], query[: self.query_maxlen - 2], [self.tokenizer.eos_token_id]]
                )
            )
            document_ids.append(
                np.concatenate(
                    [[self.tokenizer.bos_token_id], document[: self.body_maxlen - 2], [self.tokenizer.eos_token_id]]
                )
            )
            labels.append(label)

        if not self.priority_label:
            idx = np.random.permutation(len(query_ids))[: self.max_query]
        else:
            indexer = np.arange(len(query_ids))

            priority_mask = labels == self.priority_label
            priority_idx = indexer[priority_mask]

            if priority_idx.shape[0] >= self.max_query:
                priority_idx = np.random.permutation(priority_idx)[: self.max_query - 1]
                another_idx = np.random.permutation(indexer[~priority_mask])[:1]
            else:
                another_idx = np.random.permutation(indexer[~priority_mask])[: (self.max_query - priority_idx.shape[0])]

            idx = np.concatenate([priority_idx, another_idx])

        query_ids = [query_ids[i] for i in idx]
        document_ids = [document_ids[i] for i in idx]
        labels = [labels[i] for i in idx]
        if not self.priority_label:
            for _ in range(self.max_query - len(query_ids)):
                query_ids.append(self.dummy_token_id)
                document_ids.append(self.dummy_token_id)
                labels.append(self.dummy_label)
        return query_ids, document_ids, labels


def bert_multizone_concat(tokenizer, query, *document_zones):
    return np.concatenate(
        [
            [tokenizer.bos_token_id],
            query,
            *[[tokenizer.sep_token_id] if i % 2 == 0 else document_zones[i // 2] for i in range(2 * len(document_zones))],
            [tokenizer.eos_token_id],
        ]
    )


class MultizoneDataset(Dataset):
    """
    Multizone Web Dataset for 1 or more target tasks. Labels must have shape (len_dataset, label_count).
    """

    def __init__(
        self,
        queries,
        documents,
        labels,
        qids,
        tokenizer,
        max_query=16,
        max_query_len=64,
        max_texts_len=[64],
        max_len=128,
        fields=['titles'],
        mapper=None,
        token_concat_flavour=bert_multizone_concat,
        label_count=1,
        sample_count=1,
    ):
        self.token_concat_flavour = token_concat_flavour
        self.queries = queries
        self.documents = documents  # tuple of text fields
        self.labels = labels
        self.indptr = get_indptr(qids)
        self.tokenizer = tokenizer
        self.max_query = max_query
        self.max_query_len = max_query_len
        self.max_texts_len = max_texts_len
        self.max_len = max_len
        self.fields = fields
        self.dummy_token_id = self.token_concat_flavour(self.tokenizer, [], *[[] for i in range(len(fields))]).flatten()
        self.dummy_label = [-1000.0] * label_count
        self.mapper = mapper
        self.extra_tokens_num = len(self.dummy_token_id)
        self.sample_count = sample_count
        print(self.dummy_token_id, self.extra_tokens_num)

    def __len__(self):
        return len(self.indptr) - 1

    def __getitem__(self, idx):
        begidx = self.indptr[idx]
        endidx = self.indptr[idx + 1]
        indices = np.arange(begidx, endidx)
        input_ids = []
        labels = []

        for fields in zip(
            self.queries[indices], *[self.documents[field][indices] for field in self.fields], self.labels[indices]
        ):
            query, document_zones, label = fields[0], fields[1:-1], fields[-1]
            if self.mapper:
                query = np.array([self.mapper.get(i, 3) for i in query], dtype=np.int32)
                # document = np.array([self.mapper.get(i,3) for i in document], dtype=np.int32)

            query = query[: self.max_query_len]
            document_zones = list(document_zones)

            for i in range(len(document_zones)):
                document_zones[i] = document_zones[i][: self.max_texts_len[i]]
            last_zone_len = (
                self.max_len - self.extra_tokens_num - len(query) - np.sum([len(zone) for zone in document_zones[:-1]])
            )
            document_zones[-1] = document_zones[-1][: int(last_zone_len)]

            input_id = self.token_concat_flavour(self.tokenizer, query, *document_zones)
            input_ids.append(input_id)

            if isinstance(label, np.ndarray):
                labels.append(list(label))
            else:
                labels.append([label])

        labels = np.array(labels)
        zero_mask = labels == 0
        labels[zero_mask] = 1

        idxs = [
            np.random.choice(np.arange(len(labels)), min(self.max_query, len(labels)), replace=False)
            for _ in range(self.sample_count)
        ]
        stds = [np.std(labels[idx]) for idx in idxs]
        idx = idxs[np.argmax(stds)]  # samples with max variance of labels

        labels[zero_mask] = 0  # return zero label

        input_ids = [input_ids[i] for i in idx]
        labels = [labels[i] for i in idx]

        for _ in range(self.max_query - len(input_ids)):
            input_ids.append(self.dummy_token_id)
            labels.append(self.dummy_label)

        return input_ids, labels


class FlatMultizoneDataset(Dataset):
    """
    Flat Multizone Web Dataset for 1 or more target tasks. Labels must have shape (len_dataset, label_count).
    """

    def __init__(
        self,
        queries,
        documents,
        labels,
        qids,
        tokenizer,
        max_query_len=64,
        max_texts_len=[64],
        max_len=128,
        fields=['titles'],
        mapper=None,
        token_concat_flavour=bert_multizone_concat,
        add_rank_zone=False,
    ):
        self.token_concat_flavour = token_concat_flavour
        self.queries = queries
        self.documents = documents  # tuple of text fields
        self.labels = labels
        self.qids = qids
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_texts_len = max_texts_len
        self.max_len = max_len
        self.fields = fields
        self.mapper = mapper
        self.extra_tokens_num = len(
            self.token_concat_flavour(self.tokenizer, [], *[[] for i in range(len(fields))]).flatten()
        )
        # print(self.dummy_token_id, self.extra_tokens_num)
        self.add_rank_zone = add_rank_zone
        if self.add_rank_zone:
            self.max_texts_len = [1] + self.max_texts_len

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx][: self.max_query_len]
        document_zones = [self.documents[field][idx] for field in self.fields]

        if self.add_rank_zone:
            document_zones = [self.add_rank_zone] + document_zones

        if self.mapper:
            query = np.array([self.mapper.get(i, 3) for i in query], dtype=np.int32)
            # document = np.array([self.mapper.get(i,3) for i in document], dtype=np.int32)

        document_zones = list(document_zones)
        for i in range(len(document_zones)):
            document_zones[i] = document_zones[i][: self.max_texts_len[i]]
        last_zone_len = (
            self.max_len - self.extra_tokens_num - len(query) - np.sum([len(zone) for zone in document_zones[:-1]])
        )
        document_zones[-1] = document_zones[-1][: int(last_zone_len)]
        input_id = self.token_concat_flavour(self.tokenizer, query, *document_zones)

        return input_id, self.labels[idx], self.qids[idx]


class BodyPassageDataset(Dataset):
    """
    Body Dataset for 1 or more target tasks. Labels must have shape (len_dataset, label_count).
    """

    def __init__(
        self,
        queries,
        documents,
        labels,
        qids,
        tokenizer,
        max_query=16,
        max_len=289,
        max_passage=16,
        max_passage_len=225,
        stride=200,
        passage_sampling=False,
        mapper=None,
        priority_label=None,
        token_concat_flavour=bert_concat,
        label_count=1,
    ):
        self.token_concat_flavour = token_concat_flavour
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.indptr = get_indptr(qids)
        self.tokenizer = tokenizer
        self.max_query = max_query
        self.max_len = max_len
        self.dummy_token_id = self.token_concat_flavour(self.tokenizer, [], []).flatten()
        self.dummy_label = [-1000.0] * label_count
        self.mapper = mapper
        self.priority_label = priority_label

        self.extra_tokens_num = len(self.token_concat_flavour(self.tokenizer, [], []).flatten())

        self.max_passage = max_passage
        self.max_passage_len = max_passage_len
        self.stride = stride
        self.passage_sampling = passage_sampling

    def __len__(self):
        return len(self.indptr) - 1

    def __getitem__(self, idx):
        begidx = self.indptr[idx]
        endidx = self.indptr[idx + 1]
        indices = np.arange(begidx, endidx)
        input_ids = []
        passage_mask = []
        labels = []

        for query, document, label in zip(self.queries[indices], self.documents[indices], self.labels[indices]):
            if self.mapper:
                query = np.array([self.mapper.get(i, 3) for i in query], dtype=np.int32)
                document = np.array([self.mapper.get(i, 3) for i in document], dtype=np.int32)

            indexer = get_indexer(
                len(document), self.max_passage, self.max_passage_len, self.stride, self.passage_sampling
            )

            document = np.pad(
                document, (0, self.max_passage_len), 'constant', constant_values=self.tokenizer.pad_token_id
            )

            passage_ids = np.ones((self.max_passage, self.max_passage_len)) * self.tokenizer.pad_token_id
            passage_ids[: len(indexer)] = document[indexer]

            passage_mask += [1] * len(indexer) + [0] * (self.max_passage - len(indexer))

            for passage in passage_ids:
                query, passage, _ = self.tokenizer.truncate_sequences(
                    query, passage, num_tokens_to_remove=len(query) + len(passage) + self.extra_tokens_num - self.max_len
                )

                input_id = self.token_concat_flavour(self.tokenizer, query, passage)
                input_ids.append(input_id)

            if isinstance(label, np.ndarray):
                labels.append(list(label))
            else:
                labels.append([label])

        labels = np.array(labels)

        idx = np.random.choice(np.arange(len(labels)), min(self.max_query, len(labels)), replace=False)
        idx = np.sort(idx)

        input_ids = [input_ids[i * self.max_passage + k] for i in idx for k in range(self.max_passage)]
        passage_mask = [passage_mask[i * self.max_passage + k] for i in idx for k in range(self.max_passage)]
        labels = [labels[i] for i in idx]

        if not self.priority_label:
            for _ in range(self.max_passage * self.max_query - len(input_ids)):
                input_ids.append(self.dummy_token_id)
                passage_mask.append(0)

            for _ in range(self.max_query - len(labels)):
                labels.append(self.dummy_label)

        return input_ids, passage_mask, labels


class FlatPassageDataset(Dataset):
    def __init__(
        self,
        queries,
        documents,
        labels,
        qids,
        tokenizer,
        max_len=289,
        mapper=None,
        max_passage=16,
        max_passage_len=225,
        stride=200,
        passage_sampling=False,
        token_concat_flavour=bert_concat,
    ):
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.qids = qids
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.mapper = mapper

        self.token_concat_flavour = token_concat_flavour
        self.extra_tokens_num = len(self.token_concat_flavour(self.tokenizer, [], []).flatten())

        self.max_passage = max_passage
        self.max_passage_len = max_passage_len
        self.stride = stride
        self.passage_sampling = passage_sampling

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        document = self.documents[idx]
        input_ids = []

        if self.mapper:
            query = np.array([self.mapper.get(i, 3) for i in query], dtype=np.int32)
            document = np.array([self.mapper.get(i, 3) for i in document], dtype=np.int32)

        indexer = get_indexer(len(document), self.max_passage, self.max_passage_len, self.stride, self.passage_sampling)

        document = np.pad(document, (0, self.max_passage_len), 'constant', constant_values=self.tokenizer.pad_token_id)

        passage_ids = np.ones((self.max_passage, self.max_passage_len)) * self.tokenizer.pad_token_id
        passage_ids[: len(indexer)] = document[indexer]

        passage_mask = [1] * len(indexer) + [0] * (self.max_passage - len(indexer))

        for passage in passage_ids:
            query, passage, _ = self.tokenizer.truncate_sequences(
                query, passage, num_tokens_to_remove=len(query) + len(passage) + self.extra_tokens_num - self.max_len
            )

            input_id = self.token_concat_flavour(self.tokenizer, query, passage)
            input_ids.append(input_id)

        self.labels[idx]
        return input_ids, passage_mask, self.labels[idx], self.qids[idx]


class BodyPassageDatasetDSSM(Dataset):
    """
    Body Dataset for 1 or more target tasks. Labels must have shape (len_dataset, label_count).
    """

    def __init__(
        self,
        queries,
        documents,
        labels,
        qids,
        tokenizer,
        max_query=16,
        max_len=64,
        max_passage=16,
        max_passage_len=225,
        stride=200,
        passage_sampling=False,
        mapper=None,
        priority_label=None,
        label_count=1,
    ):
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.indptr = get_indptr(qids)
        self.tokenizer = tokenizer
        self.max_query = max_query
        self.max_len = max_len
        self.dummy_token_id = np.array([tokenizer.bos_token_id, tokenizer.eos_token_id])
        self.dummy_label = [-1000.0] * label_count
        self.mapper = mapper
        self.priority_label = priority_label

        self.max_passage = max_passage
        self.max_passage_len = max_passage_len
        self.stride = stride
        self.passage_sampling = passage_sampling

        self.passage_ids = np.ones((self.max_passage, self.max_passage_len - 2)).astype('int')
        self.passage_ids.fill(self.tokenizer.pad_token_id)

    def __len__(self):
        return len(self.indptr) - 1

    def __getitem__(self, idx):
        begidx = self.indptr[idx]
        endidx = self.indptr[idx + 1]
        indices = np.arange(begidx, endidx)
        query_ids = []
        document_ids = []
        labels = []

        doc_maxlen = self.stride * (self.max_passage - 1) + self.max_passage_len

        for query, document, label in zip(self.queries[indices], self.documents[indices], self.labels[indices]):
            if self.mapper:
                query = np.array([self.mapper.get(i, 3) for i in query], dtype=np.int32)
                document = np.array([self.mapper.get(i, 3) for i in document], dtype=np.int32)

            query_ids.append(
                np.concatenate([[self.tokenizer.bos_token_id], query[: self.max_len - 2], [self.tokenizer.eos_token_id]])
            )
            document_ids.append(
                np.concatenate([[self.tokenizer.bos_token_id], document[: doc_maxlen - 2], [self.tokenizer.eos_token_id]])
            )

            if isinstance(label, np.ndarray):
                labels.append(list(label))
            else:
                labels.append([label])

        labels = np.array(labels)

        idx = np.random.permutation(len(labels))[: self.max_query]

        query_ids = [query_ids[i] for i in idx]
        document_ids = [document_ids[i] for i in idx]
        labels = [labels[i] for i in idx]

        if not self.priority_label:
            for _ in range(self.max_query - len(labels)):
                labels.append(self.dummy_label)
                query_ids.append(self.dummy_token_id)
                document_ids.append(self.dummy_token_id)

        return query_ids, document_ids, labels


class FlatPassageDatasetDSSM(Dataset):
    def __init__(
        self,
        queries,
        documents,
        labels,
        qids,
        tokenizer,
        max_len=64,
        mapper=None,
        max_passage=16,
        max_passage_len=225,
        stride=200,
        passage_sampling=False,
    ):
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.qids = qids
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.mapper = mapper

        self.max_passage = max_passage
        self.max_passage_len = max_passage_len
        self.stride = stride
        self.passage_sampling = passage_sampling

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        document = self.documents[idx]

        doc_maxlen = self.stride * (self.max_passage - 1) + self.max_passage_len

        query = np.concatenate([[self.tokenizer.bos_token_id], query[: self.max_len - 2], [self.tokenizer.eos_token_id]])
        document = np.concatenate(
            [[self.tokenizer.bos_token_id], document[: doc_maxlen - 2], [self.tokenizer.eos_token_id]]
        )

        label = self.labels[idx]

        return query, document, label, self.qids[idx]


class BodyParadeDataset(IterableDataset):
    def __init__(
        self,
        train_dataset,
        tokenizer,
        max_query,
        max_len,
        max_passage=16,
        max_passage_len=225,
        stride=200,
        passage_sampling=False,
        mapper=None,
        token_concat_flavour=bert_concat,
        ranks_num=2,
        dssm_variant=False,
        step_per_epoch=900_000,
        distill_path=None,
    ):
        super().__init__()

        if dssm_variant:
            files = sorted([os.path.join(train_dataset, p) for p in os.listdir(train_dataset)])
        else:
            files = sorted(
                [
                    os.path.join(train_dataset, p, 'data.pck')
                    for p in os.listdir(train_dataset)
                    if not p.endswith(".pck") and p.startswith('train')
                ]
            )

        self.fnames = np.array(list(files))

        print(self.fnames)

        self.tokenizer = tokenizer
        self.mapper = mapper
        self.rank = None
        self.ranks_num = ranks_num

        self.max_query = max_query
        self.max_len = max_len

        self.max_passage = max_passage
        self.max_passage_len = max_passage_len
        self.stride = stride
        self.passage_sampling = passage_sampling

        self.step_per_epoch = step_per_epoch
        self.distill_path = distill_path

        self.token_concat_flavour = token_concat_flavour

        self.dssm_variant = dssm_variant
        self.custom_processor = None

    def set_rank(self, rank, ranks_num):
        self.rank = rank
        self.ranks_num = ranks_num

    @property
    def len(self):
        # return 109817 // 4
        # return 105_000 // self.ranks_num #новые ассессоры
        # return 900_000 // self.ranks_num
        # return 1_980_000 // self.ranks_num
        return self.step_per_epoch // self.ranks_num
        # return 671_000 // self.ranks_num
        # return 50_000 // self.ranks_num

    def __len__(self):
        return self.len

    def __iter__(self):
        assert self.rank is not None

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        worker_id = worker_id + self.rank * num_workers
        num_workers *= self.ranks_num

        # worker_id = self.rank
        # num_workers = self.ranks_num

        print(self.rank, worker_id, num_workers, self.dssm_variant)

        while True:
            np.random.shuffle(self.fnames)
            for fname in self.fnames:
                if self.dssm_variant:
                    print(worker_id, fname)
                    with open(os.path.join(fname, 'data.pck'), 'rb') as f:
                        train = pickle.load(f)

                    if self.distill_path:
                        if self.distill_path == 'serps':
                            print('Use serps labels')
                            labels = train['labels'].flatten()
                            labels = (50 - labels) / 50.0
                        else:
                            print('Read new labels')
                            labels = np.loadtxt(os.path.join(fname, self.distill_path))
                    else:
                        labels = train['labels'].flatten()

                    train_dataset = BodyPassageDatasetDSSM(
                        train['queries'],
                        train['bodies'],  # не забыть вернуть боди
                        labels,
                        train['qids'].flatten(),
                        self.tokenizer,
                        self.max_query,
                        self.max_len,
                        mapper=self.mapper,
                        max_passage=self.max_passage,
                        max_passage_len=self.max_passage_len,
                        stride=self.stride,
                        passage_sampling=self.passage_sampling,
                    )
                else:
                    with open(fname, 'rb') as f:
                        train = pickle.load(f)

                    train_dataset = BodyPassageDataset(
                        train['queries'],
                        train['bodies'],
                        train['labels'].flatten(),
                        train['qids'].flatten(),
                        self.tokenizer,
                        self.max_query,
                        self.max_len,
                        mapper=self.mapper,
                        token_concat_flavour=self.token_concat_flavour,
                        max_passage=self.max_passage,
                        max_passage_len=self.max_passage_len,
                        stride=self.stride,
                        passage_sampling=self.passage_sampling,
                    )

                current_len = len(train_dataset)
                index_order = np.random.permutation(current_len)

                for i in index_order:
                    if i % num_workers != worker_id:
                        continue

                    yield train_dataset[i]

            break
