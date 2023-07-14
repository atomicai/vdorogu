import string

import torch

from vdorogu.inferencer.base_container import Container
from vdorogu.nn_component.model.utils import load_by_all_means, load_state_dict_by_all_means
from vdorogu.pipeline.SOTA.web_rank.yandex_dssm.source.src.qlt_reader import Indexer
from vdorogu.pipeline.SOTA.web_rank.yandex_dssm.source.src.utils import EuclideanMatcher


class YandexModel(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, sentence_dim):
        super().__init__()

        self.embed = torch.nn.Embedding(1000002, embedding_dim, sparse=True, dtype=torch.float32)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, sentence_dim),
        )

    def forward(self, x):
        x = self.embed(x)
        x = torch.max(x, dim=-2)[0]
        x = self.ff(x)
        return x


def remove_punctuation(s):
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)
    return s


class YandexDSSM(Container):
    def __init__(self, hparams):
        super().__init__(modes=("scores", "query_emb", "document_emb"))

        self.max_token_len = 20
        self.embedding_dim = 512
        self.hidden_dim = 256
        self.sentence_dim = 64

        self.checkpoint_path = "epoch=16-step=4045778.ckpt"
        self.known_words_path = "unigram_1mil.txt"

    def load(self):
        self.matcher = EuclideanMatcher()
        self.indexer = Indexer(self.known_words_path)
        self.model = YandexModel(self.embedding_dim, self.hidden_dim, self.sentence_dim)
        ckpt = load_by_all_means(self.checkpoint_path)
        load_state_dict_by_all_means(self.model, ckpt)

    def prepare_data(self, *fields):
        query, document = "", ""

        if self.mode == "scores":
            query, document = fields
        elif self.mode == "query_emb":
            (query,) = fields
        elif self.mode == "document_emb":
            (document,) = fields

        query = remove_punctuation(query)
        document = remove_punctuation(document)

        q = self.indexer(query.split())[: self.max_token_len]
        t = self.indexer(document.split())[: self.max_token_len]
        q = q + [0] * (self.max_token_len - len(q))
        t = t + [0] * (self.max_token_len - len(t))
        return q, t

    def collate(self, batch):
        queries, titles = map(list, zip(*batch))
        return torch.tensor(queries), torch.tensor(titles)

    def forward(self, batch):
        queries, titles = batch
        if self.mode == "scores":
            queries_emb = self.model.forward(queries)
            titles_emb = self.model.forward(titles)
            y_hat = self.matcher(queries_emb, titles_emb)
            return y_hat
        elif self.mode == "query_emb":
            queries_emb = self.model.forward(queries)
            return queries_emb
        elif self.mode == "document_emb":
            titles_emb = self.model.forward(titles)
            return titles_emb
