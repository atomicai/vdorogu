from collections import OrderedDict

import numpy as np
import torch
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer

from vdorogu.dataset.web import bert_concat, valid_collate, valid_collate_dssm
from vdorogu.inferencer.base_container import Container, match_arcifact_path
from vdorogu.nn_component.model.nlp.bert_dssm import Bert, BertDSSM, BertSentenceEmbDSSM
from vdorogu.nn_component.model.utils import load_by_all_means, load_state_dict_by_all_means


class BertDSSMContainer(Container):
    def __init__(self, hparams):
        super().__init__(modes=("scores", "query_emb", "document_emb"))

        hparams.setdefault("query_maxlen", 64)
        hparams.setdefault("document_maxlen", 64)

        hparams.setdefault("tokenizer_path", "xlm-roberta-base")

        self.tokenizer_path = match_arcifact_path(hparams, "tokenizer_path")
        self.config_path = match_arcifact_path(hparams, "config_path")
        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "weights.pck")

        self.query_maxlen = hparams['query_maxlen']
        self.document_maxlen = hparams['document_maxlen']

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

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        self.model = BertDSSM(self.config_path, self.checkpoint_path)

    def tokenize_(self, text, maxlen):
        ids = self.tokenizer.encode(" ".join(text.lower().split()[:maxlen]), add_special_tokens=False)
        return np.concatenate([[self.tokenizer.bos_token_id], ids[: maxlen - 2], [self.tokenizer.eos_token_id]])

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
            return valid_collate_dssm(batch, pad8=False)
        else:
            return valid_collate_dssm(batch)

    def forward(self, batch):
        if self.mode == "scores":
            q, d = batch
            return (self.model(q, is_query=True) * self.model(d, is_query=False)).sum(-1) * 10
        elif self.mode == "query_emb":
            (q,) = batch
            return self.model(q, is_query=True)
        elif self.mode == "document_emb":
            (d,) = batch
            return self.model(d, is_query=False)

    def debug(self, batch):
        q, d = batch

        log = []

        q_emb = self.model(q, is_query=True)
        d_emb = self.model(d, is_query=False)

        res = (q_emb * d_emb).sum(-1) * 10

        log.append(('q_emb', q_emb))
        log.append(('d_emb', d_emb))
        log.append(('res', res))

        return OrderedDict(log)


class BertDSSMwithNormsContainer(Container):
    def __init__(self, hparams):
        super().__init__(modes=("scores", "query_emb", "document_emb"))

        hparams.setdefault("query_maxlen", 64)
        hparams.setdefault("document_maxlen", 64)

        hparams.setdefault("tokenizer_path", "xlm-roberta-base")

        self.tokenizer_path = match_arcifact_path(hparams, "tokenizer_path")
        self.config_path = match_arcifact_path(hparams, "config_path")
        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "weights.pck")

        self.query_maxlen = hparams['query_maxlen']
        self.document_maxlen = hparams['document_maxlen']

    def sample_input(self):
        if self.mode == "scores":
            return self._n_fields_sample_input(2)
        elif self.mode == "query_emb":
            return self._n_fields_sample_input(1)
        elif self.mode == "document_emb":
            return self._n_fields_sample_input(1)

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        self.model = BertDSSM(self.config_path, self.checkpoint_path)

    def tokenize_(self, text, maxlen):
        ids = self.tokenizer.encode(" ".join(text.lower().split()[:maxlen]), add_special_tokens=False)
        return np.concatenate([[self.tokenizer.bos_token_id], ids[: maxlen - 2], [self.tokenizer.eos_token_id]])

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
        return valid_collate_dssm(batch)

    def forward_doc(self, d, is_query=False):
        return self.model(d, is_query)

    def forward_query(self, q, is_query=True):
        return self.model(q, is_query)

    def forward(self, batch):
        q, d, _, _ = batch

        if self.mode == "scores":
            q_embed, q_norms = self.forward_query(q, is_query=True)
            doc_embed, doc_norms = self.forward_doc(d, is_query=False)
            scores = (q_embed * doc_embed).sum(-1) * 10

            return torch.cat((scores.view(-1, 1), q_norms, doc_norms), dim=-1)

        elif self.mode == "query_emb":
            return self.model(q, is_query=True)
        elif self.mode == "document_emb":
            return self.model(d, is_query=False)


class RuBertTiny2DSSMContrainer(BertDSSMContainer):
    def __init__(self, hparams):
        Container.__init__(self, modes=("scores", "query_emb", "document_emb"))
        self.tokenizer_path = match_arcifact_path(hparams, "tokenizer_path", "tokenizer")
        self.config_path = match_arcifact_path(hparams, "config_path", "config.json")
        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "weights.pth")
        self.query_maxlen = hparams['query_maxlen']
        self.document_maxlen = hparams['document_maxlen']

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = BertSentenceEmbDSSM("bert", self.config_path, checkpoint_path=self.checkpoint_path)
        self.model.pad_token_id = self.tokenizer.pad_token_id

    def collate(self, batch):
        if self.optimized_for == 'onnx':  # (cpu, gpu, onnx)
            return valid_collate_dssm(batch, pad_token_id=self.tokenizer.pad_token_id, pad8=False)
        else:
            return valid_collate_dssm(batch, pad_token_id=self.tokenizer.pad_token_id)


class BertContainer(Container):
    def __init__(self, hparams):
        super().__init__()

        hparams.setdefault("model_class", "xlmroberta")

        hparams.setdefault("query_maxlen", 64)
        hparams.setdefault("document_maxlen", 64)

        hparams.setdefault("tokenizer_path", "xlm-roberta-base")

        self.model_class = hparams['model_class']

        self.tokenizer_path = match_arcifact_path(hparams, "tokenizer_path")
        self.config_path = match_arcifact_path(hparams, "config_path")
        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "weights.pck")

        self.query_maxlen = hparams['query_maxlen']
        self.document_maxlen = hparams['document_maxlen']

        self.num_labels = 1
        self.pad_id = 1

        self.mapper_path = None
        self.mapper = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        self.model = Bert(self.model_class, self.config_path, config_path=self.config_path, pretrained=False)

        ckpt = load_by_all_means(self.checkpoint_path)
        load_state_dict_by_all_means(self.model, ckpt)

        if self.mapper_path:
            self.mapper = torch.load(self.mapper_path)

        self.model.pad_token_id = self.pad_id

    def prepare_data(self, query, document):
        query = self.tokenizer.encode(query.lower(), add_special_tokens=False)
        document = self.tokenizer.encode(" ".join(document.lower().split()[: self.document_maxlen]), add_special_tokens=False)

        if self.mapper:
            if self.mapper:
                query = np.array([self.mapper.get(i, 3) for i in query], dtype=np.int32)
                document = np.array([self.mapper.get(i, 3) for i in document], dtype=np.int32)

        ids = bert_concat(self.tokenizer, query[: self.query_maxlen], document[: self.document_maxlen])

        return ids, 0.0, 1  # fake fields for collate

    def collate(self, batch):
        return valid_collate(batch, self.pad_id)

    def forward(self, batch):
        input_ids, _, _ = batch
        return self.model(input_ids)


class SberBertContainer(BertContainer):
    def load(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)

        self.tokenizer.bos_token = '[CLS]'
        special_tokens_dict = {'eos_token': '[EOS]'}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model = Bert(self.model_class, self.config_path, config_path=self.config_path, pretrained=False)
        self.model.bert.resize_token_embeddings(len(self.tokenizer))

        ckpt = load_by_all_means(self.checkpoint_path)
        load_state_dict_by_all_means(self.model, ckpt)

        if self.mapper_path:
            self.mapper = torch.load(self.mapper_path)

        self.model.pad_token_id = self.tokenizer.pad_token_id
        self.pad_id = self.tokenizer.pad_token_id
