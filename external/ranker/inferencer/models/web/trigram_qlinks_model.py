from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vdorogu.dataset.web import pad_sequence
from vdorogu.inferencer.base_container import Container, match_arcifact_path

TEXT_MAXLEN = 20
WORDS_MAXLEN = 30


def dot(a, b):
    return torch.matmul(a.unsqueeze(-2), b.unsqueeze(-1)).squeeze(-1)


def norm(x, p=2, dim=-1):
    norm = x.norm(p=p, dim=dim, keepdim=True)
    return x.div(norm)


class ProdModel(nn.Module):
    def __init__(self, embedding_size, scale_train=False):
        super().__init__()

        self.scale_train = scale_train

        self.embedding = nn.Embedding(29438, 128, sparse=True)

        self.word_conv = nn.Conv2d(128, 128, (1, 5), stride=(1, 1), padding=(0, 2))

        self.embed_projector = nn.Linear(128, embedding_size)

        self.activation = F.elu

        if self.scale_train:
            self.scale = nn.Linear(1, 1)

    def forward_half(self, input):
        emb = self.embedding(input)
        # (batch_size, word_i, trigram_i, emb_size)

        # make channels_first
        emb = emb.permute(0, 3, 1, 2)

        emb = self.word_conv(emb)
        emb = self.activation(emb)

        # (batch_size, emb_size, word_i, trigram_i)
        emb = emb.view(emb.size()[:2] + (-1,)).max(-1)[0]

        emb = self.embed_projector(emb)

        return emb

    def forward(self, query, title):
        query_emb = self.forward_half(query)
        title_emb = self.forward_half(title)

        query_emb = norm(query_emb)
        title_emb = norm(title_emb)
        scores = dot(query_emb, title_emb)

        if self.training and self.scale_train:
            scores = self.scale(scores)

        return scores


class ProdModel128(ProdModel):
    def __init__(self):
        super().__init__(128)


def load_trigrams(fname):
    trigrams = {}
    with open(fname, "r", encoding="utf-8") as fin:
        for line in fin:
            num, gram = line.strip().split("\t")
            trigrams[gram] = int(num)

    return trigrams


def w_to_ngrams(w, n=3):
    w = "#%s#" % w
    for i in range(len(w) - n + 1):
        yield w[i : i + n]


def tokenize(trigrams, text, max_words, max_trigrams):
    res = []

    words = list(text.upper().split())
    if len(words) < 1:
        words += [""] * 3

    for w in words:
        if len(res) >= max_words:
            break

        parts = []
        for t in w_to_ngrams(w):
            if len(parts) >= max_trigrams:
                break

            parts.append(trigrams.get(t, len(trigrams)))
        while len(parts) < max_trigrams:
            parts.append(len(trigrams))

        res.append(parts)

    return np.array(res)


class TrigramQlinksModelContainer(Container):
    def __init__(self, hparams):
        super().__init__(modes=("scores", "query_emb", "document_emb"))

        hparams.setdefault("text_maxlen", TEXT_MAXLEN)
        hparams.setdefault("words_maxlen", WORDS_MAXLEN)

        self.trigrams_path = match_arcifact_path(hparams, "trigrams_path", "trigrams.txt")
        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "weights.pck")

        self.text_maxlen = hparams["text_maxlen"]
        self.words_maxlen = hparams["words_maxlen"]

    def sample_input(self):
        if self.mode == "scores":
            return self._n_fields_sample_input(2)
        elif self.mode == "query_emb":
            return self._n_fields_sample_input(1)
        elif self.mode == "document_emb":
            return self._n_fields_sample_input(1)

    def load(self):
        self.trigrams = load_trigrams(self.trigrams_path)

        self.model = ProdModel128()
        self.model.load_state_dict(torch.load(self.checkpoint_path))

    def prepare_data(self, *fields):
        query, document = "", ""

        if self.mode == "scores":
            query, document = fields
        elif self.mode == "query_emb":
            (query,) = fields
        elif self.mode == "document_emb":
            (document,) = fields

        query = tokenize(self.trigrams, query, self.text_maxlen, self.words_maxlen)
        document = tokenize(self.trigrams, document, self.text_maxlen, self.words_maxlen)

        return query, document

    def collate(self, batch):
        inputs = list(zip(*batch))

        pad_id = len(self.trigrams)

        q = [torch.from_numpy(x) for x in inputs[0]]
        d = [torch.from_numpy(x) for x in inputs[1]]

        q = pad_sequence(q, padding_value=pad_id)
        d = pad_sequence(d, padding_value=pad_id)

        return q, d

    def forward(self, batch):
        q, d = batch

        if self.mode == "scores":
            return self.model(q, d)
        elif self.mode == "query_emb":
            return norm(self.model.forward_half(q))
        elif self.mode == "document_emb":
            return norm(self.model.forward_half(d))

    def debug(self, batch):
        q, d = batch

        log = []

        q_emb = self.model.forward_half(q)
        d_emb = self.model.forward_half(d)

        log.append(("q_emb_unnormed", q_emb))
        log.append(("d_emb_unnormed", d_emb))

        query_emb = norm(q_emb)
        title_emb = norm(d_emb)

        res = dot(query_emb, title_emb)

        log.append(("res", res))

        return OrderedDict(log)


Container = TrigramQlinksModelContainer
