import math
import re
import unicodedata
from collections import defaultdict
from itertools import chain

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset, Sampler


class RawTextDataset(IterableDataset):
    def __init__(
        self,
        filename,
        tokenizer,
        preprocessing,
        max_len=128,
        separator='\t',
        mapper=None,
        actual_fields=(0, 1, 2),
        max_len_list=None,
    ):
        self.filename = filename
        self.tokenizer = tokenizer
        self.preprocessing = preprocessing

        self.mapper = lambda x: x
        if mapper:
            self.mapper = lambda x: mapper.get(x, 3)

        self.separator = separator
        self.actual_fields = actual_fields

        self.max_len = max_len
        self.max_len_list = max_len_list

    def __iter__(self):
        with open(self.filename, 'r') as f:
            for line in f:
                datum = line.strip().split(self.separator)
                datum = np.array(datum)[np.array(self.actual_fields)]

                tokenized_items = []

                for item in datum:
                    tokenized_items.append(
                        [
                            self.mapper(i)
                            for i in tokenizer.encode(
                                self.preprocessing(item), max_length=self.max_len, add_special_tokens=False, truncation=True
                            )
                        ]
                    )

                yield tokenized_items


def bert_concat(tokenizer, query, document):
    return np.concatenate(
        [[tokenizer.bos_token_id], query, [tokenizer.sep_token_id, tokenizer.sep_token_id], document, [tokenizer.eos_token_id]]
    )


def roberta_concat(tokenizer, query, document):
    return np.concatenate([[0], query, [2, 2], document, [2]])


def sentence_emb_concat(tokenizer, query, document):
    return np.concatenate([[tokenizer.cls_token_id], query, [tokenizer.sep_token_id], document])


def mbert_concat(tokenizer, query, document):
    return np.concatenate(
        [
            [tokenizer.cls_token_id],
            query,
            [tokenizer.sep_token_id],
            document,
            [tokenizer.sep_token_id],
        ]
    )


def gpt_concat(tokenizer, query, document):
    return np.concatenate([[tokenizer.bos_token_id], query, [tokenizer.sep_token_id], document, [tokenizer.eos_token_id]])


def gpt_concat_wo_tokens(tokenizer, query, document):
    return np.concatenate([query, document])


def t5_concat(tokenizer, query, document):
    return np.concatenate(
        [[tokenizer.bos_token_id], query, [tokenizer.eos_token_id, tokenizer.eos_token_id], document, [tokenizer.eos_token_id]]
    )


def t5_ptuning_concat(tokenizer, query, document, promt_lenght=[3, 3, 3]):
    return np.concatenate([[-1] * (promt_lenght[0] - 1) + [-2], query, [-2] * promt_lenght[1], document, [-2] * promt_lenght[2]])


class QueryDataset(Dataset):
    """
    Query Dataset for 1 or more target tasks. Labels must have shape (len_dataset, label_count).
    """

    def __init__(
        self,
        queries,
        documents,
        labels,
        qids,
        tokenizer,
        max_query=16,
        max_len=128,
        mapper=None,
        priority_label=None,
        token_concat_flavour=bert_concat,
        label_count=1,
        dummy_label=-1000.0,
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
        self.dummy_label = [dummy_label] * label_count
        self.mapper = mapper
        self.priority_label = priority_label

        self.extra_tokens_num = len(self.token_concat_flavour(self.tokenizer, [], []).flatten())

    def __len__(self):
        return len(self.indptr) - 1

    def __getitem__(self, idx):
        begidx = self.indptr[idx]
        endidx = self.indptr[idx + 1]
        indices = np.arange(begidx, endidx)
        input_ids = []
        labels = []

        for query, document, label in zip(self.queries[indices], self.documents[indices], self.labels[indices]):
            if self.mapper:
                query = np.array([self.mapper.get(i, 3) for i in query], dtype=np.int32)
                document = np.array([self.mapper.get(i, 3) for i in document], dtype=np.int32)
            query, document, _ = self.tokenizer.truncate_sequences(
                query, document, num_tokens_to_remove=len(query) + len(document) + self.extra_tokens_num - self.max_len
            )
            input_id = self.token_concat_flavour(self.tokenizer, query, document)
            input_ids.append(input_id)

            if isinstance(label, np.ndarray):
                labels.append(list(label))
            else:
                labels.append([label])

        labels = np.array(labels)
        if not self.priority_label:
            idx = np.random.permutation(len(labels))[: self.max_query]
        else:
            indexer = np.arange(len(labels))

            priority_mask = labels == self.priority_label
            priority_idx = indexer[priority_mask]

            if len(priority_idx) >= self.max_query:
                priority_idx = np.random.permutation(priority_idx)[: self.max_query - 1]
                another_idx = np.random.permutation(indexer[~priority_mask])[:1]
            else:
                another_idx = np.random.permutation(indexer[~priority_mask])[: (self.max_query - priority_idx.shape[0])]

            idx = np.concatenate([priority_idx, another_idx])

        input_ids = [input_ids[i] for i in idx]
        labels = [labels[i] for i in idx]
        if not self.priority_label:
            for _ in range(self.max_query - len(input_ids)):
                input_ids.append(self.dummy_token_id)
                labels.append(self.dummy_label)

        return input_ids, labels


class QueryDatasetWithNegativeSampling(Dataset):
    def __init__(
        self,
        queries,
        documents,
        labels,
        qids,
        tokenizer,
        max_query=16,
        max_len=128,
        mapper=None,
        positive_label_mask=None,
        p=0.2,
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
        self.positive_label_mask = positive_label_mask
        self.p = p

        self.extra_tokens_num = len(self.token_concat_flavour(self.tokenizer, [], []).flatten())

    def __len__(self):
        return len(self.indptr) - 1

    def __getitem__(self, idx):
        begidx = self.indptr[idx]
        endidx = self.indptr[idx + 1]

        positive_label_count = np.sum(self.positive_label_mask[begidx:endidx])
        input_ids = []
        labels = []

        indices = np.arange(begidx, begidx + positive_label_count)
        size = min(int(self.p * len(indices)) + 1, endidx - (begidx + positive_label_count))
        if size > 0:
            indices_neg = np.random.choice(np.arange(begidx + positive_label_count, endidx), size=size, replace=False)
            indices = np.concatenate((indices, indices_neg))

        for query, document, label in zip(self.queries[indices], self.documents[indices], self.labels[indices]):
            if self.mapper:
                query = np.array([self.mapper.get(i, 3) for i in query], dtype=np.int32)
                document = np.array([self.mapper.get(i, 3) for i in document], dtype=np.int32)
            query, document, _ = self.tokenizer.truncate_sequences(
                query, document, num_tokens_to_remove=len(query) + len(document) + self.extra_tokens_num - self.max_len
            )
            input_id = self.token_concat_flavour(self.tokenizer, query, document)
            input_ids.append(input_id)

            if isinstance(label, np.ndarray):
                labels.append(list(label))
            else:
                labels.append([label])

        labels = np.array(labels)
        idx = np.random.permutation(len(labels))[: self.max_query]

        input_ids = [input_ids[i] for i in idx]
        labels = [labels[i] for i in idx]

        for _ in range(self.max_query - len(input_ids)):
            input_ids.append(self.dummy_token_id)
            labels.append(self.dummy_label)

        return input_ids, labels


class FlatDataset(Dataset):
    def __init__(self, queries, documents, labels, qids, tokenizer, max_len=128, mapper=None, token_concat_flavour=bert_concat):
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.qids = qids
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.mapper = mapper

        self.token_concat_flavour = token_concat_flavour
        self.extra_tokens_num = len(self.token_concat_flavour(self.tokenizer, [], []).flatten())

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx][: self.max_len // 2]
        document = self.documents[idx]
        if self.mapper:
            query = np.array([self.mapper.get(i, 3) for i in query], dtype=np.int32)
            document = np.array([self.mapper.get(i, 3) for i in document], dtype=np.int32)

        query, document, _ = self.tokenizer.truncate_sequences(
            query, document, num_tokens_to_remove=len(query) + len(document) + self.extra_tokens_num - self.max_len
        )

        input_id = self.token_concat_flavour(self.tokenizer, query, document)
        label = self.labels[idx]
        return input_id, self.labels[idx], self.qids[idx]


class FlatDatasetWithEmb(Dataset):
    def __init__(
        self, queries, documents, labels, qids, tokenizer, query_emb, max_len=128, mapper=None, token_concat_flavour=bert_concat
    ):
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.qids = qids
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.mapper = mapper
        self.query_emb = query_emb

        self.token_concat_flavour = token_concat_flavour
        self.extra_tokens_num = len(self.token_concat_flavour(self.tokenizer, [], []).flatten())

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx][: self.max_len // 2]
        document = self.documents[idx]
        qid = self.qids[idx]

        query, document, _ = self.tokenizer.truncate_sequences(
            query, document, num_tokens_to_remove=len(query) + len(document) + self.extra_tokens_num - self.max_len
        )

        input_id = self.token_concat_flavour(self.tokenizer, query, document)
        label = self.labels[idx]
        return self.query_emb[qid], input_id, label, qid


class FlatDatasetWithoutLabel(Dataset):
    def __init__(self, queries, documents, tokenizer, token_concat_flavour=bert_concat):
        self.queries = queries
        self.documents = documents
        self.tokenizer = tokenizer
        self.token_concat_flavour = token_concat_flavour

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        document = self.documents[idx]

        input_id = self.token_concat_flavour(self.tokenizer, query, document)
        return input_id


class QueryDatasetDSSM(Dataset):
    def __init__(
        self,
        queries,
        documents,
        labels,
        qids,
        tokenizer,
        max_query=16,
        max_len=128,
        mapper=None,
        priority_label=None,
        valid_mode=False,
        dummy_label=-1000.0,
    ):
        '''
        priority_label - the label that must be sampled
        '''
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.qids = qids
        self.indptr = get_indptr(qids)
        self.tokenizer = tokenizer
        self.max_query = max_query
        self.max_len = max_len
        self.dummy_token_id = np.array([tokenizer.bos_token_id, tokenizer.eos_token_id])
        self.dummy_label = dummy_label
        self.mapper = mapper
        self.priority_label = priority_label
        self.valid_mode = valid_mode

    def __len__(self):
        return len(self.indptr) - 1

    def __getitem__(self, idx):
        begidx = self.indptr[idx]
        endidx = self.indptr[idx + 1]
        indices = np.arange(begidx, endidx)
        query_ids = []
        document_ids = []
        labels = []
        qids = []

        for query, document, label, qid in zip(
            self.queries[indices], self.documents[indices], self.labels[indices], self.qids[indices]
        ):
            if self.mapper:
                query = np.array([self.mapper.get(i, 3) for i in query], dtype=np.int32)
                document = np.array([self.mapper.get(i, 3) for i in document], dtype=np.int32)

            query_ids.append(
                np.concatenate([[self.tokenizer.bos_token_id], query[: self.max_len - 2], [self.tokenizer.eos_token_id]])
            )
            document_ids.append(
                np.concatenate([[self.tokenizer.bos_token_id], document[: self.max_len - 2], [self.tokenizer.eos_token_id]])
            )
            labels.append(label)
            qids.append(qid)

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
        qids = [qids[i] for i in idx]

        if not self.priority_label:
            for _ in range(self.max_query - len(query_ids)):
                query_ids.append(self.dummy_token_id)
                document_ids.append(self.dummy_token_id)
                labels.append(self.dummy_label)
                qids.append(self.dummy_label)

        if not self.valid_mode:
            return query_ids, document_ids, labels
        else:
            return query_ids, document_ids, labels, qids


class QueryDatasetDSSMWithNegativeSampling(Dataset):
    def __init__(
        self,
        queries,
        documents,
        labels,
        qids,
        tokenizer,
        max_query=16,
        max_len=128,
        mapper=None,
        positive_label_mask=None,
        p=0.2,
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
        self.dummy_label = -1000.0
        self.mapper = mapper
        self.positive_label_mask = positive_label_mask
        self.p = p

        self.extra_tokens_num = len(self.token_concat_flavour(self.tokenizer, [], []).flatten())

    def __len__(self):
        return len(self.indptr) - 1

    def __getitem__(self, idx):
        begidx = self.indptr[idx]
        endidx = self.indptr[idx + 1]
        positive_label_count = np.sum(self.positive_label_mask[begidx:endidx])

        query_ids = []
        document_ids = []
        labels = []

        indices = np.arange(begidx, begidx + positive_label_count)
        size = min(int(self.p * len(indices)) + 1, endidx - (begidx + positive_label_count))
        if size > 0:
            indices_neg = np.random.choice(np.arange(begidx + positive_label_count, endidx), size=size, replace=False)
            indices = np.concatenate((indices, indices_neg))

        for query, document, label in zip(self.queries[indices], self.documents[indices], self.labels[indices]):
            if self.mapper:
                query = np.array([self.mapper.get(i, 3) for i in query], dtype=np.int32)
                document = np.array([self.mapper.get(i, 3) for i in document], dtype=np.int32)

            query_ids.append(
                np.concatenate([[self.tokenizer.bos_token_id], query[: self.max_len - 2], [self.tokenizer.eos_token_id]])
            )
            document_ids.append(
                np.concatenate([[self.tokenizer.bos_token_id], document[: self.max_len - 2], [self.tokenizer.eos_token_id]])
            )
            labels.append(label)

        labels = np.array(labels)

        idx = np.random.permutation(len(labels))[: self.max_query]

        query_ids = [query_ids[i] for i in idx]
        document_ids = [document_ids[i] for i in idx]
        labels = [labels[i] for i in idx]

        for _ in range(self.max_query - len(query_ids)):
            query_ids.append(self.dummy_token_id)
            document_ids.append(self.dummy_token_id)
            labels.append(self.dummy_label)

        return query_ids, document_ids, labels


class FlatDatasetDSSM(Dataset):
    def __init__(self, queries, documents, labels, qids, tokenizer, max_len=64, mapper=None):
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.qids = qids
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mapper = mapper

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx][: self.max_len - 2]
        document = self.documents[idx][: self.max_len - 2]
        if self.mapper:
            query = np.array([self.mapper.get(i, 3) for i in query], dtype=np.int32)
            document = np.array([self.mapper.get(i, 3) for i in document], dtype=np.int32)
        query = np.concatenate([[self.tokenizer.bos_token_id], query, [self.tokenizer.eos_token_id]])
        document = np.concatenate([[self.tokenizer.bos_token_id], document, [self.tokenizer.eos_token_id]])
        return query, document, self.labels[idx], self.qids[idx]


class FlatDatasetDSSMWithEmbedding(Dataset):
    def __init__(self, queries, documents, labels, qids, tokenizer, max_len=64, mapper=None):
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.qids = qids
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mapper = mapper

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        qid = self.qids[idx]
        query = self.queries[qid]
        document = self.documents[idx][: self.max_len - 2]
        if self.mapper:
            query = np.array([self.mapper.get(i, 3) for i in query], dtype=np.int32)
            document = np.array([self.mapper.get(i, 3) for i in document], dtype=np.int32)
        document = np.concatenate([[self.tokenizer.bos_token_id], document, [self.tokenizer.eos_token_id]])
        return query, document, self.labels[idx], qid


class RankPairPermutationDatasetTrain(Dataset):
    def __init__(self, queries, documents, labels, qids, tokenizer, max_len=192, mapper=None):
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.indptr = get_indptr(qids)
        self.tokenizer = tokenizer

        self.max_len = max_len
        self.dummy_token_id = np.array([tokenizer.bos_token_id, tokenizer.eos_token_id])
        self.dummy_label = -1000.0
        self.mapper = mapper

        self.label_dict = {
            (0, 1): 0,
            (1, 0): 1,
        }
        self.dummy_seq = np.array([2, 2])

    def __len__(self):
        return len(self.indptr) - 1

    def __getitem__(self, idx):
        endidx = self.indptr[idx + 1]
        begidx = self.indptr[idx]

        indices = np.arange(begidx, endidx)

        queries = np.array([i for i in self.queries[indices]])
        docs = np.array([i for i in self.documents[indices]])
        labels = np.array([i for i in self.labels[indices]])

        indexes = np.arange(docs.shape[0])

        try_hard = 0
        for i in range(10):
            idx = np.random.choice(indexes, 2)
            _labels = labels[idx]
            if np.abs((_labels - _labels[::-1])).sum() > 0:
                break
            try_hard += 1

        _q = queries[0]
        _docs = docs[idx]
        _labels = labels[idx]

        objects, labels = [], []

        if try_hard == 9:
            # добиваем dummy_token
            _docs = [_docs[0], self.dummy_seq]
            _labels = np.array([1, 0])

        revert_doc = _docs[::-1]
        permutation_label = self.label_dict[tuple(_labels.argsort())]

        _0 = np.array([0])
        _2_2 = np.array([2, 2])
        _2 = np.array([2])

        labels.append(permutation_label)
        objects.append(np.concatenate([_0, _q[:64], _2_2, _docs[0][:64], _2_2, _docs[1][:64], _2]))

        permutation_label = self.label_dict[tuple(_labels[::-1].argsort())]
        labels.append(permutation_label)
        objects.append(np.concatenate([_0, _q[:64], _2_2, revert_doc[0][:64], _2_2, revert_doc[1][:64], _2]))

        return objects, labels


class RankPairPermutationDatasetValid(Dataset):
    def __init__(
        self,
        queries,
        documents,
        labels,
        qids,
        tokenizer,
        max_len=64,
        mapper=None,
    ):
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.indptr = get_indptr(qids)
        self.tokenizer = tokenizer
        self.qids = qids

        self.max_len = max_len
        self.dummy_token_id = np.array([tokenizer.eos_token_id, tokenizer.eos_token_id])
        self.dummy_label = -1000.0
        self.mapper = mapper

    def __len__(self):
        return len(self.indptr) - 1

    def __getitem__(self, idx):
        endidx = self.indptr[idx + 1]
        begidx = self.indptr[idx]

        indices = np.arange(begidx, endidx)
        resh = np.arange(len(indices))
        np.random.shuffle(resh)
        indices = indices[resh]

        queries = np.array([i[: self.max_len] for i in self.queries[indices]])
        docs = [np.array(i)[: self.max_len] for i in self.documents[indices]]
        labels = [np.array(i) for i in self.labels[indices]]

        return queries, docs, labels, self.qids[indices]

    @staticmethod
    def pack_q_a_b(a, b, query):
        query = query.squeeze().long()
        d = query.device
        _0 = torch.LongTensor([0]).to(d)
        _2_2 = torch.LongTensor([2, 2]).to(d)
        _2 = torch.LongTensor([2]).to(d)
        return (
            torch.cat([_0, query, _2_2, a, _2_2, b, _2]).reshape(1, -1),
            torch.cat([_0, query, _2_2, b, _2_2, a, _2]).reshape(1, -1),
        )

    @staticmethod
    def no_collate(batch, pad_token_id=1):
        inputs = list(zip(*batch))
        input_queries_ids = pad8_sequence([torch.from_numpy(x) for x in inputs[0]], padding_value=pad_token_id).long()

        input_documents_ids = [torch.from_numpy(x).long() for x in inputs[1][0]]
        labels = torch.from_numpy(np.stack(inputs[2])).float()
        qids = torch.from_numpy(np.stack(inputs[3])).float()

        return input_queries_ids, input_documents_ids, labels, qids


class QueryDatasetMultitargetWithNegative(Dataset):
    def __init__(
        self,
        queries,
        documents,
        labels,
        qids,
        tokenizer,
        max_query=16,
        max_len=128,
        mapper=None,
        priority_label=None,
        is_dssm=False,
        dummy_label=-1000.0,
        negative='default',
        min_length=3,
        token_concat_flavour=bert_concat,
        use_emb=False,
    ):
        self.queries = queries
        self.documents = documents
        self.documents_length = len(self.documents)
        self.labels = labels
        self.qids = qids
        self.indptr = get_indptr(qids)
        self.tokenizer = tokenizer
        self.max_query = max_query
        self.max_len = max_len
        self.dummy_token_id = np.array(
            [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.eos_token_id, tokenizer.eos_token_id]
        )
        self.dummy_label = dummy_label
        self.mapper = mapper
        self.priority_label = priority_label
        self.is_dssm = is_dssm
        self.negative = negative
        self.min_length = min_length
        self.token_concat_flavour = token_concat_flavour
        self.use_emb = use_emb

    def __len__(self):
        return len(self.indptr) - 1

    def __getitem__(self, idx):
        begidx = self.indptr[idx]
        endidx = self.indptr[idx + 1]
        indices = np.arange(begidx, endidx)
        input_ids = []
        labels = []
        query_list = []
        document_list = []

        if not self.use_emb:
            quers = self.queries[indices]
        else:
            quers = []
            for qid in self.qids[indices]:
                quers.append(self.queries[qid])

        for query, document, label in zip(quers, self.documents[indices], self.labels[indices]):
            if self.mapper:
                query = np.array([self.mapper.get(i, 3) for i in query], dtype=np.int32)
                document = np.array([self.mapper.get(i, 3) for i in document], dtype=np.int32)

            if len(query) <= self.min_length or len(document) <= self.min_length:
                continue

            if not self.is_dssm:
                query, document, _ = self.tokenizer.truncate_sequences(
                    query, document, num_tokens_to_remove=len(query) + len(document) + 4 - self.max_len
                )
                input_id = self.token_concat_flavour(self.tokenizer, query, document)[: self.max_len]
                input_ids.append(np.array(input_id))
            else:
                if self.use_emb:
                    query_list.append(query)
                else:
                    query_list.append(
                        np.concatenate([[self.tokenizer.bos_token_id], query[: self.max_len - 2], [self.tokenizer.eos_token_id]])
                    )
                document_list.append(
                    np.concatenate([[self.tokenizer.bos_token_id], document[: self.max_len - 2], [self.tokenizer.eos_token_id]])
                )
            labels.append(label)

        labels = np.array(labels)
        labels_size = self.labels.shape[1]
        if not self.priority_label:
            if self.negative == 'half':
                idx = np.random.permutation(len(labels))[: self.max_query // 2]
            else:
                idx = np.random.permutation(len(labels))[: self.max_query]

        else:
            indexer = np.arange(len(labels))

            priority_mask = labels == self.priority_label
            priority_idx = indexer[priority_mask]

            if len(priority_idx) >= self.max_query:
                priority_idx = np.random.permutation(priority_idx)[: self.max_query - 1]
                another_idx = np.random.permutation(indexer[~priority_mask])[:1]
            else:
                another_idx = np.random.permutation(indexer[~priority_mask])[: (self.max_query - priority_idx.shape[0])]

            idx = np.concatenate([priority_idx, another_idx])

        if not self.is_dssm:
            input_ids = [input_ids[i] for i in idx]
        else:
            query_list = [query_list[i] for i in idx]
            document_list = [document_list[i] for i in idx]

        labels = [labels[i] for i in idx]
        positive_labels = [1 for i in labels]
        # negative mining
        if not self.priority_label:
            if self.negative == 'default':
                negatives_count = self.max_query - len(input_ids)
            elif self.negative == 'free':
                negatives_count = self.max_query - len(labels)
            else:
                negatives_count = self.max_query - len(labels)

            for _ in range(negatives_count):
                neg_idx = np.random.randint(0, self.documents_length)

                if not self.is_dssm:
                    input_id = np.concatenate(
                        [
                            [self.tokenizer.bos_token_id],
                            query,
                            [
                                self.tokenizer.eos_token_id,
                                self.tokenizer.eos_token_id,
                            ],
                            self.documents[neg_idx],
                            [self.tokenizer.eos_token_id],
                        ]
                    )
                    input_ids.append(input_id[: self.max_len])
                else:
                    if self.use_emb:
                        query_list.append(query)
                    else:
                        query_list.append(
                            np.concatenate(
                                [[self.tokenizer.bos_token_id], query[: self.max_len - 2], [self.tokenizer.eos_token_id]]
                            )
                        )
                    document_list.append(
                        np.concatenate(
                            [
                                [self.tokenizer.bos_token_id],
                                self.documents[neg_idx][: self.max_len - 2],
                                [self.tokenizer.eos_token_id],
                            ]
                        )
                    )
                labels.append(np.array([self.dummy_label for i in range(labels_size)]))
                positive_labels.append(0)

        labels = np.concatenate([np.array(labels), np.array(positive_labels).reshape(-1, 1)], axis=1)
        if not self.is_dssm:
            return input_ids, labels

        return query_list, document_list, labels


class QueryOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, queries, labels, tokenizer, max_len=256):
        self.queries = queries
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.qids = np.arange(self.queries.shape[0])

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = self.queries[index][: self.max_len - 2]
        query = np.concatenate([[self.tokenizer.bos_token_id], query, [self.tokenizer.eos_token_id]])
        return query, self.labels[index], self.qids[index]


class QueryHardNegativeDataset(Dataset):
    def __init__(
        self,
        queries,
        documents,
        labels,
        qids,
        negative_queries,
        negative_documents,
        negative_labels,
        negative_qids,
        tokenizer,
        max_query=16,
        max_negative_query=8,
        max_len=128,
        is_dssm=False,
        dummy_label=-1000.0,
        negative='free',
        min_length=0,
        use_emb=False,
    ):
        self.dataset = QueryDatasetMultitargetWithNegative(
            queries=queries,
            documents=documents,
            labels=labels,
            qids=qids,
            tokenizer=tokenizer,
            max_query=max_query,
            max_len=max_len,
            is_dssm=is_dssm,
            negative=negative,
            dummy_label=dummy_label,
            min_length=min_length,
            use_emb=use_emb,
        )

        self.negative_dataset = QueryDatasetMultitargetWithNegative(
            queries=negative_queries,
            documents=negative_documents,
            labels=negative_labels,
            qids=negative_qids,
            dummy_label=0.0,
            tokenizer=tokenizer,
            max_query=max_negative_query,
            is_dssm=is_dssm,
            negative='free',
            min_length=min_length,
            use_emb=use_emb,
        )
        self.is_dssm = is_dssm

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if not self.is_dssm:
            input_ids, labels = self.dataset[idx]
            negative_input_ids, negative_labels = self.negative_dataset[idx]
            input_ids.extend(negative_input_ids)
            negative_labels[:, 1] = 0
            labels = np.concatenate((labels, negative_labels))
            return input_ids, labels

        query_list, document_list, labels = self.dataset[idx]
        negative_query_list, negative_document_list, negative_labels = self.negative_dataset[idx]
        query_list.extend(negative_query_list)
        document_list.extend(negative_document_list)
        negative_labels[:, 1] = 0
        labels = np.concatenate([labels, negative_labels])
        return query_list, document_list, labels


def get_indptr(column):
    group = []
    k = -1
    prev = None
    for x in column:
        if x != prev:
            k += 1
            prev = x
        group.append(k)
    group = np.cumsum(np.append(0, np.bincount(group)))
    return group


def get_groups(column):
    group = []
    k = -1
    prev = None
    for x in column:
        if x != prev:
            k += 1
            prev = x
        group.append(k)
    return np.array(group)


class BucketBatchSampler(Sampler):
    # want inputs to be an array
    def __init__(self, length, bins=[32, 64, 128, 196, 256], batch_sizes=[128, 64, 32, 20, 16, 8], shuffle=True):
        self.batch_sizes = batch_sizes
        self.batch_size = batch_sizes[0]
        self.drop_last = False
        self.bins = bins
        self.batch_map = defaultdict(list)
        self.shuffle = shuffle

        buckets = np.digitize(length, bins)
        for idx, bucket in enumerate(buckets):
            self.batch_map[bucket].append(idx)
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        resid_list = []
        batch_list = []
        for bucket_idx, indices in sorted(self.batch_map.items(), key=lambda x: x[0]):
            batch_size = self.batch_sizes[bucket_idx]
            np.random.shuffle(indices)
            for group in [indices[i : (i + batch_size)] for i in range(0, len(indices), batch_size)]:
                if len(group) == batch_size:
                    batch_list.append(group)
                else:
                    resid_list += group
        for group in [resid_list[i : (i + self.batch_sizes[-1])] for i in range(0, len(resid_list), self.batch_sizes[-1])]:
            batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self):
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)
        # shuffle all the batches so they arent ordered by bucket size
        if self.shuffle:
            np.random.shuffle(self.batch_list)
        for i in self.batch_list:
            yield i


class BucketBatchSyncSampler(Sampler):
    # want inputs to be an array
    def __init__(self, length, bins=[32, 64, 128, 196, 256], batch_sizes=[128, 64, 32, 20, 16, 8], shuffle=True, world_size=4):
        self.batch_sizes = batch_sizes
        self.bins = bins
        self.batch_map = defaultdict(list)
        self.shuffle = shuffle
        self.world_size = world_size
        self.resid_key = 10000
        buckets = np.digitize(length, bins)
        for idx, bucket in enumerate(buckets):
            self.batch_map[bucket].append(idx)
        self.batch_dict = self._generate_batch_map()
        self.num_batches = sum([len(x) for x in self.batch_dict.values()])

    def _generate_batch_map(self):
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        resid_list = []
        batch_dict = {i: [] for i in range(len(self.bins) + 1)}
        batch_dict[self.resid_key] = []
        for bucket_idx, indices in sorted(self.batch_map.items(), key=lambda x: x[0]):
            batch_size = self.batch_sizes[bucket_idx]
            np.random.shuffle(indices)
            for group in [indices[i : (i + batch_size)] for i in range(0, len(indices), batch_size)]:
                if len(group) == batch_size:
                    batch_dict[bucket_idx].append(group)
                else:
                    resid_list += group
        for group in [resid_list[i : (i + self.batch_sizes[-1])] for i in range(0, len(resid_list), self.batch_sizes[-1])]:
            batch_dict[self.resid_key].append(group)
        return batch_dict

    def __len__(self):
        return self.num_batches

    def batch_count(self):
        return self.num_batches

    def __iter__(self):
        self.batch_dict = self._generate_batch_map()
        self.num_batches = sum([len(x) for x in self.batch_dict.values()])
        lens = torch.tensor([len(self.batch_dict[x]) for x in sorted(self.batch_dict.keys())]).cuda()
        gather_lens = [torch.zeros_like(lens) for _ in range(self.world_size)]
        torch.distributed.all_gather(gather_lens, lens)
        lens = torch.stack(gather_lens).min(0)[0].cpu().numpy()
        indices = []
        for i, x in zip(sorted(self.batch_dict.keys()), lens):
            indices.append(np.ones(x, dtype=np.int64) * i)
        indices = np.concatenate(indices)
        np.random.seed(42)
        np.random.shuffle(indices)
        for ind in indices:
            yield self.batch_dict[ind].pop()
        for x in chain.from_iterable(self.batch_dict.values()):
            yield x


def pad_sequence(sequences, batch_first=True, padding_value=0.0, max_len=None):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    trailing_dims = sequences[0].size()[1:]

    max_len = max_len or max([s.size(0) for s in sequences])

    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)

    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def pad8_sequence(sequences, batch_first=True, padding_value=0.0, max_len=None):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]

    trailing_dims = sequences[0].size()[1:]
    max_len = max_len or int(np.ceil(max([s.size(0) for s in sequences]) / 8) * 8)
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)

    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def pad_collate(inputs, pad_token_id=1):
    inputs = list(zip(*inputs))
    input_ids = [torch.from_numpy(x) for x in chain.from_iterable(inputs[0])]
    input_ids = pad8_sequence(input_ids, padding_value=pad_token_id).long()
    labels = torch.from_numpy(np.array(list(chain.from_iterable(inputs[1])))).float()
    return input_ids, labels


def valid_collate(inputs, pad_token_id=1):
    inputs = list(zip(*inputs))
    input_ids = pad8_sequence([torch.from_numpy(x) for x in inputs[0]], padding_value=pad_token_id).long()
    labels = torch.from_numpy(np.stack(inputs[1])).float()
    queries = torch.from_numpy(np.stack(inputs[2])).long()
    return input_ids, labels, queries


def valid_collate_query_only(inputs, pad_token_id=1):
    inputs = list(zip(*inputs))
    queries = pad8_sequence([torch.from_numpy(x) for x in inputs[0]], padding_value=pad_token_id).long()
    labels = torch.from_numpy(np.stack(inputs[1])).float()
    return queries, labels


def valid_collate_with_emb(inputs, pad_token_id=1):
    inputs = list(zip(*inputs))
    input_query_emb = [torch.from_numpy(x) for x in inputs[0]]
    input_query_emb = pad8_sequence(input_query_emb, padding_value=pad_token_id).float()
    input_ids = pad8_sequence([torch.from_numpy(x) for x in inputs[1]], padding_value=pad_token_id).long()
    labels = torch.from_numpy(np.stack(inputs[2])).float()
    queries = torch.from_numpy(np.stack(inputs[3])).long()
    return input_query_emb, input_ids, labels, queries


def pad_collate_neg(inputs):
    return pad_collate(inputs, -1)


def valid_collate_neg(inputs):
    return valid_collate(inputs, -1)


def pad_collate_dssm(inputs, pad_token_id=1):
    inputs = list(zip(*inputs))
    input_queries_ids = [torch.from_numpy(x) for x in chain.from_iterable(inputs[0])]
    input_documents_ids = [torch.from_numpy(x) for x in chain.from_iterable(inputs[1])]
    input_queries_ids = pad8_sequence(input_queries_ids, padding_value=pad_token_id).long()
    input_documents_ids = pad8_sequence(input_documents_ids, padding_value=pad_token_id).long()
    labels = torch.from_numpy(np.array(list(chain.from_iterable(inputs[2])))).float()
    return input_queries_ids, input_documents_ids, labels


def pad_collate_dssm_emb(inputs, pad_token_id=1):
    inputs = list(zip(*inputs))
    input_queries_ids = [torch.from_numpy(x) for x in chain.from_iterable(inputs[0])]
    input_documents_ids = [torch.from_numpy(x) for x in chain.from_iterable(inputs[1])]
    input_queries_ids = pad8_sequence(input_queries_ids, padding_value=pad_token_id).float()
    input_documents_ids = pad8_sequence(input_documents_ids, padding_value=pad_token_id).long()
    labels = torch.from_numpy(np.array(list(chain.from_iterable(inputs[2])))).float()
    return input_queries_ids, input_documents_ids, labels


def valid_collate_dssm(inputs, pad_token_id=1, pad8=True):
    inputs = list(zip(*inputs))
    if pad8:
        input_queries_ids = pad8_sequence([torch.from_numpy(x) for x in inputs[0]], padding_value=pad_token_id).long()
        input_documents_ids = pad8_sequence([torch.from_numpy(x) for x in inputs[1]], padding_value=pad_token_id).long()
    else:
        input_queries_ids = pad_sequence([torch.from_numpy(x) for x in inputs[0]], padding_value=pad_token_id).long()
        input_documents_ids = pad_sequence([torch.from_numpy(x) for x in inputs[1]], padding_value=pad_token_id).long()

    labels = torch.from_numpy(np.stack(inputs[2])).float()
    qids = torch.from_numpy(np.stack(inputs[3])).long()
    return input_queries_ids, input_documents_ids, labels, qids


def valid_collate_dssm_emb(inputs, pad_token_id=1):
    inputs = list(zip(*inputs))
    input_queries_ids = pad8_sequence([torch.from_numpy(x) for x in inputs[0]], padding_value=pad_token_id).float()
    input_documents_ids = pad8_sequence([torch.from_numpy(x) for x in inputs[1]], padding_value=pad_token_id).long()
    labels = torch.from_numpy(np.stack(inputs[2])).float()
    qids = torch.from_numpy(np.stack(inputs[3])).long()
    return input_queries_ids, input_documents_ids, labels, qids


def pad_collate_dssm_neg(inputs):
    return pad_collate_dssm(inputs, -1)


def valid_collate_dssm_neg(inputs):
    return valid_collate_dssm(inputs, -1)


class MixedDataset(Dataset):
    def __init__(self, main_dataset, mix_in_dataset, mix_proba=0.0):
        self.main_dataset = main_dataset
        self.mix_in_dataset = mix_in_dataset
        self.mix_proba = mix_proba

    def __len__(self):
        return len(self.main_dataset)

    def __getitem__(self, idx):
        if np.random.random() > self.mix_proba:
            return self.main_dataset[idx]
        else:
            rand_idx_low = len(self.mix_in_dataset) * idx // len(self.main_dataset)
            rand_idx_high = math.ceil(len(self.mix_in_dataset) * (idx + 1) / len(self.main_dataset))
            rand_idx = np.random.randint(rand_idx_low, rand_idx_high)
            return self.mix_in_dataset[rand_idx]
