import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm as BertLayerNorm
from torch.nn.functional import gelu
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

# from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids
# from transformers.models.bert.modeling_bert import BertEmbeddings
from transformers import (
    AlbertConfig,
    AlbertModel,
    AlbertTokenizer,
    BartConfig,
    BartModel,
    BartTokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
    ElectraConfig,
    ElectraModel,
    ElectraTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForMaskedLM,
    XLMRobertaModel,
    XLMRobertaTokenizer,
    XLNetConfig,
    XLNetModel,
    XLNetTokenizer,
)

from external.ranker.nn_component.layer.noise import GaussianNoise
from external.ranker.nn_component.layer.pooling import MaxPool2D as MaxPool
from external.ranker.nn_component.model.utils import load_by_all_means, load_state_dict_by_all_means

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetModel, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer),
    "electra": (ElectraConfig, ElectraModel, ElectraTokenizer),
    "bart": (BartConfig, BartModel, BartTokenizer),
}


class BertDSSM(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config_path, checkpoint_path=None, proj_size=64):
        super().__init__()
        conf = XLMRobertaConfig.from_pretrained(config_path)
        self.bert = XLMRobertaModel(conf)
        self.proj = nn.Linear(conf.hidden_size, proj_size)
        self.proj.weight.data.normal_(0, 0.02)
        self.bert.embeddings.token_type_embeddings.weight.data.normal_(0, 0.02)
        self.pad_token_id = 1
        self.max_pool = MaxPool(batch_first=True)

        if checkpoint_path is not None:
            ckpt = load_by_all_means(checkpoint_path)
            try:
                load_state_dict_by_all_means(self, ckpt)
            except RuntimeError:
                load_state_dict_by_all_means(self.bert, ckpt)
                print("Warning: initing only bert part of BertDSSM model", file=sys.stderr)

    def forward(self, inputs, is_query=True):
        token_type_ids = torch.zeros_like(inputs) if is_query else torch.ones_like(inputs)
        attention_mask = inputs != self.pad_token_id

        if self.pad_token_id < 0:
            inputs[~attention_mask] = 0

        x = self.bert(inputs, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        x = self.max_pool(x, attention_mask)
        x = self.proj(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class BertDSSMWithNorms(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config_path, checkpoint_path=None, proj_size=64):
        super().__init__()
        conf = XLMRobertaConfig.from_pretrained(config_path)
        self.bert = XLMRobertaModel(conf)
        self.proj = nn.Linear(conf.hidden_size, proj_size)
        self.proj.weight.data.normal_(0, 0.02)
        self.bert.embeddings.token_type_embeddings.weight.data.normal_(0, 0.02)
        self.pad_token_id = 1
        self.max_pool = MaxPool(batch_first=True)

        if checkpoint_path is not None:
            ckpt = load_by_all_means(checkpoint_path)
            try:
                load_state_dict_by_all_means(self, ckpt)
            except RuntimeError:
                load_state_dict_by_all_means(self.bert, ckpt)
                print("Warning: initing only bert part of BertDSSM model", file=sys.stderr)

    def forward(self, inputs, is_query=True, return_norms=False):
        token_type_ids = torch.zeros_like(inputs) if is_query else torch.ones_like(inputs)
        attention_mask = inputs != self.pad_token_id

        if self.pad_token_id < 0:
            inputs[~attention_mask] = 0

        x = self.bert(inputs, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)[0]

        x = x.last_hidden_state
        x = self.max_pool(x, attention_mask)

        norms = torch.linalg.norm(x, dim=-1)

        x = self.proj(x)
        x = F.normalize(x, p=2, dim=-1)

        if return_norms:
            return x, norms
        else:
            return x


class BertDSSMFixedEmb(BertDSSM):
    def __init__(self, config_path, checkpoint_path=None, proj_size=64):
        super().__init__(config_path, checkpoint_path, proj_size)

        # for param in self.bert.embeddings.word_embeddings.parameters():
        #    param.requires_grad = False
        for param in self.bert.embeddings.position_embeddings.parameters():
            param.requires_grad = False
        # for param in self.bert.embeddings.token_type_embeddings.parameters():
        #    param.requires_grad = False
        try:
            from vdorogu.nn_component.model.nlp.gpt_dssm import EmbeddingClip
        except ImportError:
            EmbeddingClip = None

        self.clip = EmbeddingClip

    def forward(self, inputs, is_query=True):
        token_type_ids = torch.zeros_like(inputs) if is_query else torch.ones_like(inputs)
        attention_mask = inputs != self.pad_token_id

        if self.pad_token_id < 0:
            inputs[~attention_mask] = 0

        inputs_embeds = self.bert.embeddings.word_embeddings(inputs)

        if self.training:
            inputs_embeds = self.clip.apply(inputs_embeds, inputs < self.bert.config.vocab_size - 1)

        x = self.bert(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        x = self.max_pool(x, attention_mask)
        x = self.proj(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class BertSmall(nn.Module):
    def __init__(self, config_path, checkpoint_path=None):
        super().__init__()
        conf = XLMRobertaConfig.from_pretrained(config_path)
        self.bert = XLMRobertaModel(conf)

        self.head = nn.Linear(conf.hidden_size, 1)
        self.head.weight.data.normal_(0, 0.02)

        self.bert.embeddings.token_type_embeddings.weight.data.normal_(0, 0.02)

        self.pad_token_id = 1
        self.max_pool = MaxPool(batch_first=True)

        if checkpoint_path is not None:
            ckpt = load_by_all_means(checkpoint_path)
            try:
                load_state_dict_by_all_means(self, ckpt)
            except RuntimeError:
                load_state_dict_by_all_means(self.bert, ckpt)
                print("Warning: initing only bert part of BertSmall model", file=sys.stderr)

    def forward(self, inputs):
        attention_mask = inputs != self.pad_token_id
        x = self.bert(inputs, attention_mask=attention_mask)[0][:, 0]
        # x = self.max_pool(x, attention_mask)
        x = self.head(x)
        return x


class Bert(nn.Module):
    def __init__(self, model_class, model_path, config_path=None, pretrained=True, output_size=1):
        super().__init__()
        if config_path is None:
            config_path = os.path.join(model_path, 'config.json')

        # self.conf = MODEL_CLASSES[model_class][0].from_pretrained(config_path)
        self.conf = MODEL_CLASSES[model_class][0].from_pretrained(model_path)

        if pretrained:
            self.bert = MODEL_CLASSES[model_class][1].from_pretrained(model_path, config=self.conf)
        else:
            self.bert = MODEL_CLASSES[model_class][1](config=self.conf)
        self.head = nn.Linear(self.conf.hidden_size, output_size)
        self.head.weight.data.normal_(0, 0.02)
        self.drop = nn.Dropout(0.1)
        for param in self.bert.embeddings.word_embeddings.parameters():
            param.requires_grad = False
        for param in self.bert.embeddings.position_embeddings.parameters():
            param.requires_grad = False
        for param in self.bert.embeddings.token_type_embeddings.parameters():
            param.requires_grad = False
        self.pad_token_id = 1

    def forward(self, input_ids):
        attention_mask = input_ids != self.pad_token_id
        x = self.bert(input_ids, attention_mask=attention_mask)[0][:, 0]
        x = self.drop(x)
        outputs = self.head(x)
        return outputs


class BertLargeDSSM(BertDSSM):
    '''
    DSSM with complex projection layers
    '''

    def __init__(self, config_path, proj_size=64, n_head=1):
        super().__init__(config_path=config_path)
        conf = XLMRobertaConfig.from_pretrained(config_path)
        self.bert = XLMRobertaModel(conf)
        self.proj_size = proj_size

        assert proj_size < 128
        self.n_head = n_head
        self.proj = nn.Sequential(
            nn.LayerNorm(conf.hidden_size),
            GaussianNoise(0.1),
            nn.Linear(conf.hidden_size, 512),
            nn.PReLU(),
            nn.LayerNorm(512),
            GaussianNoise(0.1),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.LayerNorm(256),
            GaussianNoise(0.1),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, proj_size),
            nn.PReLU(),
        )  # после нормализуем приводя к сфере -- используем в других задачах это представление

        self.multihead_layer = nn.Sequential(
            nn.Linear(proj_size, self.n_head * proj_size),
            nn.PReLU(),
        )

        self.bert.embeddings.token_type_embeddings.weight.data.normal_(0, 0.02)
        self.pad_token_id = 1
        self.max_pool = MaxPool(batch_first=True)

    def forward(self, inputs, is_query=True):
        token_type_ids = torch.zeros_like(inputs) if is_query else torch.ones_like(inputs)
        attention_mask = inputs != self.pad_token_id
        if self.pad_token_id < 0:
            inputs[~attention_mask] = 0

        x = self.bert(inputs, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]

        x = self.max_pool(x, attention_mask)
        x = self.proj(x)
        _x = F.normalize(x, p=2, dim=-1)
        x = self.multihead_layer(_x)

        return [_x] + [
            F.normalize(x[:, self.proj_size * i : self.proj_size * (i + 1)], p=2, dim=-1) for i in range(self.n_head)
        ]


class BertSentenceEmb(nn.Module):
    def __init__(self, model_class, config_path, checkpoint_path=None, pretrained=False, output_size=1):
        super().__init__()

        Config, Model, Tokenizer = MODEL_CLASSES[model_class]

        if pretrained:
            self.model = Model.from_pretrained(config_path)
        else:
            conf = Config.from_pretrained(config_path)
            self.model = Model(conf)

        self.conf = self.model.config

        self.head = nn.Linear(self.conf.hidden_size, output_size)
        self.head.weight.data.normal_(0, 0.02)

        self.pad_token_id = 1

        if checkpoint_path is not None:
            ckpt = load_by_all_means(checkpoint_path)
            try:
                load_state_dict_by_all_means(self, ckpt)
            except RuntimeError:
                load_state_dict_by_all_means(self.model, ckpt)
                print("Warning: initing only base_model part of T5 model", file=sys.stderr)

    def forward(self, input_ids):
        attention_mask = input_ids != self.pad_token_id

        x = self.model(input_ids, attention_mask=attention_mask).last_hidden_state

        x = x[:, 0]
        # x = (x * attention_mask.unsqueeze(-1).float()).sum(1) / (attention_mask.float().sum(-1)).clamp(min=1).unsqueeze(-1)

        outputs = self.head(x)
        return outputs


class BertSentenceEmbDSSM(nn.Module):
    def __init__(self, model_class, config_path, checkpoint_path=None, pretrained=False, output_size=64):
        super().__init__()

        Config, Model, Tokenizer = MODEL_CLASSES[model_class]

        if pretrained:
            self.model = Model.from_pretrained(config_path)
        else:
            conf = Config.from_pretrained(config_path)
            self.model = Model(conf)

        self.conf = self.model.config

        self.head = nn.Linear(self.conf.hidden_size, output_size)
        self.head.weight.data.normal_(0, 0.02)

        self.pad_token_id = 1

        if checkpoint_path is not None:
            ckpt = load_by_all_means(checkpoint_path)
            try:
                load_state_dict_by_all_means(self, ckpt)
            except RuntimeError:
                load_state_dict_by_all_means(self.model, ckpt)
                print("Warning: initing only base_model part of T5 model", file=sys.stderr)

    def forward(self, input_ids, is_query=True):
        token_type_ids = torch.zeros_like(input_ids) if is_query else torch.ones_like(input_ids)

        attention_mask = input_ids != self.pad_token_id

        x = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).last_hidden_state

        x = x[:, 0]
        x = self.head(x)
        x = F.normalize(x, p=2, dim=-1)

        return x
