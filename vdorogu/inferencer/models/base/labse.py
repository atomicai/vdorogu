from vdorogu.inferencer.base_container import Container, match_arcifact_path

from transformers import AutoTokenizer
from vdorogu.nn_component.model.nlp.bert_dssm import BertSentenceEmb

from vdorogu.dataset.web import valid_collate, valid_collate_dssm, sentence_emb_concat
from vdorogu.nn_component.model.utils import load_by_all_means, load_state_dict_by_all_means

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F


class LaBSEContainer(Container):
    def __init__(self, hparams):
        super().__init__()

        hparams.setdefault("query_maxlen", 64)
        hparams.setdefault("document_maxlen", 64)

        self.tokenizer_path = match_arcifact_path(hparams, "tokenizer_path")
        self.config_path = match_arcifact_path(hparams, "config_path")
        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "weights.pck")

        self.query_maxlen = hparams['query_maxlen']
        self.document_maxlen = hparams['document_maxlen']

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = BertSentenceEmb("bert", self.config_path, pretrained=False)

        ckpt = load_by_all_means(self.checkpoint_path)
        load_state_dict_by_all_means(self.model, ckpt)

        self.model.pad_token_id = self.tokenizer.pad_token_id
        self.pad_id = self.tokenizer.pad_token_id

    def prepare_data(self, query, document):
        query = self.tokenizer.encode(query.lower(), add_special_tokens=False)
        document = self.tokenizer.encode(document.lower(), add_special_tokens=False)

        ids = sentence_emb_concat(self.tokenizer, query[: self.query_maxlen], document[: self.document_maxlen])

        return ids, 0.0, 1  # fake fields for collate

    def collate(self, batch):
        return valid_collate(batch, self.pad_id)

    def forward(self, batch):
        input_ids, _, _ = batch
        return self.model(input_ids)


class LaBSEDSSMContainer(Container):
    def __init__(self, hparams):
        super().__init__(modes=("scores", "query_emb", "document_emb"))

        hparams.setdefault("query_maxlen", 64)
        hparams.setdefault("document_maxlen", 64)

        self.tokenizer_path = match_arcifact_path(hparams, "tokenizer_path")
        self.config_path = match_arcifact_path(hparams, "config_path")
        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "weights.pck")

        self.query_maxlen = hparams['query_maxlen']
        self.document_maxlen = hparams['document_maxlen']

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = BertSentenceEmb("bert", self.config_path, pretrained=False, output_size=512)
        special_tokens_dict = {'eos_token': '[EOS]'}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer.bos_token = self.tokenizer.cls_token
        self.model.model.resize_token_embeddings(len(self.tokenizer))

        ckpt = load_by_all_means(self.checkpoint_path)
        load_state_dict_by_all_means(self.model, ckpt)

        self.model.pad_token_id = self.tokenizer.pad_token_id
        self.pad_id = self.tokenizer.pad_token_id

    def tokenize_(self, text, maxlen):
        ids = self.tokenizer.encode(" ".join(text.lower().split()[:maxlen]), add_special_tokens=False)
        return np.concatenate([[self.tokenizer.bos_token_id], ids[: maxlen - 2], [self.tokenizer.sep_token_id]])

    def sample_input(self):
        if self.mode == "scores":
            return self._n_fields_sample_input(2)
        elif self.mode == "query_emb":
            return self._n_fields_sample_input(1)
        elif self.mode == "document_emb":
            return self._n_fields_sample_input(1)

    def process_mode_batch(self, batch):
        if self.mode == "scores":
            return [batch[0], batch[1]]
        elif self.mode == "query_emb":
            return [batch[0]]
        elif self.mode == "document_emb":
            return [batch[1]]

    def prepare_data(self, *fields):
        query, document = "", ""

        if self.mode == "scores":
            query, document = fields
        elif self.mode == "query_emb":
            (query,) = fields
        elif self.mode == "document_emb":
            (document,) = fields

        query = self.tokenize_(query, self.query_maxlen)
        document = self.tokenize_(document, self.document_maxlen)

        return query, document, 0.0, 1  # fake fields for collate

    def collate(self, batch):
        if self.optimized_for == 'onnx':  # (cpu, gpu, onnx)
            return valid_collate_dssm(batch, pad_token_id=self.pad_id, pad8=False)
        else:
            return valid_collate_dssm(batch, pad_token_id=self.pad_id)

    def forward(self, batch):
        if self.mode == "scores":
            q, d = batch
            query_emb = self.model(q)
            query_emb = F.normalize(query_emb, p=2, dim=-1)
            document_emb = self.model(d)
            document_emb = F.normalize(document_emb, p=2, dim=-1)

            return (query_emb * document_emb).sum(-1) * 10
        elif self.mode == "query_emb":
            (q,) = batch
            query_emb = self.model(q)
            query_emb = F.normalize(query_emb, p=2, dim=-1)

            return query_emb
        elif self.mode == "document_emb":
            (d,) = batch
            document_emb = self.model(d)
            document_emb = F.normalize(document_emb, p=2, dim=-1)

            return document_emb

    def debug(self, batch):
        q, d = batch
        log = []

        query_emb = self.model(q)
        query_emb = F.normalize(query_emb, p=2, dim=-1)
        document_emb = self.model(d)
        document_emb = F.normalize(document_emb, p=2, dim=-1)

        res = (query_emb * document_emb).sum(-1) * 10

        log.append(('q_emb', query_emb[:, :10]))
        log.append(('d_emb', document_emb[:, :10]))
        log.append(('res', res))

        return OrderedDict(log)
