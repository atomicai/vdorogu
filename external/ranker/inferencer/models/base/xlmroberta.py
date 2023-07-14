import numpy as np
import torch
from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification

from vdorogu.dataset.web import bert_concat, valid_collate
from vdorogu.inferencer.base_container import Container, match_arcifact_path
from vdorogu.nn_component.model.utils import load_by_all_means, load_state_dict_by_all_means


class XlmrobertaContainer(Container):
    def __init__(self, hparams):
        super().__init__()

        hparams.setdefault("query_maxlen", 64)
        hparams.setdefault("document_maxlen", 64)

        hparams.setdefault("tokenizer_path", "xlm-roberta-base")

        self.tokenizer_path = match_arcifact_path(hparams, "tokenizer_path")
        self.config_path = match_arcifact_path(hparams, "config_path")
        self.checkpoint_path = match_arcifact_path(hparams, "checkpoint_path", "weights.pck")

        self.query_maxlen = hparams["query_maxlen"]
        self.document_maxlen = hparams["document_maxlen"]

        self.num_labels = 1
        self.pad_id = 1

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        conf = XLMRobertaConfig.from_pretrained(self.config_path, num_labels=self.num_labels)
        self.model = XLMRobertaForSequenceClassification(conf)

        ckpt = load_by_all_means(self.checkpoint_path)
        load_state_dict_by_all_means(self.model, ckpt)

    def prepare_data(self, query, document):
        query = self.tokenizer.encode(query.lower(), add_special_tokens=False)
        document = self.tokenizer.encode(document.lower(), add_special_tokens=False)

        ids = bert_concat(self.tokenizer, query[: self.query_maxlen], document[: self.document_maxlen])

        return ids, 0.0, 1  # fake fields for collate

    def collate(self, batch):
        return valid_collate(batch, self.pad_id)

    def forward(self, batch):
        input_ids, _, _ = batch
        return self.model(input_ids, attention_mask=(input_ids != self.pad_id)).logits
