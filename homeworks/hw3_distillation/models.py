from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig, BertForTokenClassification, set_seed
from transformers.models.bert.modeling_bert import BertLayer

from .data import BERT_NAME, label_names


id2label = dict(enumerate(label_names))
label2id = {tag: i for i, tag in id2label.items()}


class FactorizedEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        rank = config.embedding_rank
        self.embedding = nn.Embedding(
            config.vocab_size,
            rank,
            padding_idx=config.pad_token_id
        )
        self.projection = nn.Linear(rank, config.hidden_size, bias=False)

    def forward(self, input_ids):
        return self.projection(self.embedding(input_ids))

    @torch.no_grad()
    def init_svd(self, weight):
        rank = self.embedding.embedding_dim
        u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
        sqrt_s = s[:rank].sqrt()
        embedding_weight = u[:, :rank] * sqrt_s
        projection_weight = vh[:rank].T * sqrt_s

        self.embedding.weight.copy_(embedding_weight)
        self.projection.weight.copy_(projection_weight)

class LoRA(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, width))
        self.b = nn.Parameter(torch.zeros(width, rank))
        nn.init.kaiming_uniform_(self.a, a=sqrt(5))

    def forward(self, x):
        return F.linear(F.linear(x, self.a), self.b)


class LoRALinear(nn.Module):
    def __init__(self, linear, lora):
        super().__init__()
        self.linear = linear
        self.lora = lora

    def forward(self, x):
        return self.linear(x) + self.lora(x)

class SharedLayers(nn.Module):
    _tied_weights_keys = [r'blocks\.']  # kostil для чекпоинтов

    def __init__(self, layer, config, rank=0):
        super().__init__()
        self.layer = layer
        self.depth = config.num_hidden_layers
        self.rank = rank
        if rank:
            width = layer.attention.self.query.in_features
            self.query = nn.ModuleList(LoRA(width, rank) for _ in range(self.depth))
            self.value = nn.ModuleList(LoRA(width, rank) for _ in range(self.depth))
            self.blocks = nn.ModuleList(
                self.lora_layer(config, query, value)
                for query, value in zip(self.query, self.value)
            )

    def __iter__(self):
        if not self.rank:
            return iter([self.layer] * self.depth)
        return iter(self.blocks)

    def lora_layer(self, config, query, value):
        shared = self.layer
        base = shared.attention.self
        layer = BertLayer(config)
        layer.intermediate = shared.intermediate
        layer.output = shared.output
        layer.attention.output = shared.attention.output

        attn = layer.attention.self
        attn.query = LoRALinear(base.query, query)
        attn.key = base.key
        attn.value = LoRALinear(base.value, value)
        return layer

class BertForNER(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        if getattr(config, 'embedding_rank', None):
            embeddings = FactorizedEmbedding(config)
            embeddings.apply(self._init_weights)
            self.set_input_embeddings(embeddings)
        if getattr(config, 'shared', False):
            layers = self.bert.encoder.layer
            self.bert.encoder.layer = SharedLayers(
                layers[0],
                config,
                getattr(config, 'lora_rank', 0)
            )
        teacher_width = getattr(config, 'teacher_width', None)
        if teacher_width:
            self.hidden_projection = nn.Sequential(
                nn.Linear(config.hidden_size, 128, bias=False),
                nn.Linear(128, teacher_width, bias=False)
            )
            self.hidden_projection.apply(self._init_weights)

def load_teacher(seed=42):
    set_seed(seed)
    return BertForNER.from_pretrained(BERT_NAME, id2label=id2label, label2id=label2id)

def add_lora(checkpoint, rank=8, seed=42):
    set_seed(seed)
    model = BertForNER.from_pretrained(checkpoint).requires_grad_(False)
    layer = model.bert.encoder.layer.layer
    model.bert.encoder.layer = SharedLayers(
        layer,
        model.config,
        rank
    ).train(model.training)
    model.config.lora_rank = rank
    return model


@torch.no_grad()
def factorize(checkpoint, rank=64, seed=42):
    set_seed(seed)
    model = BertForNER.from_pretrained(checkpoint)
    model.config.embedding_rank = rank
    embeddings = FactorizedEmbedding(model.config)
    embeddings.init_svd(model.get_input_embeddings().weight)
    model.set_input_embeddings(embeddings)
    return model


def make_student(tokenizer, seed=42, shared=False, teacher_width=None):
    set_seed(seed)
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        hidden_size=512,
        num_hidden_layers=5,
        num_attention_heads=8,
        intermediate_size=2048,
        embedding_rank=128,
        id2label=id2label,
        label2id=label2id,
        shared=shared,
        teacher_width=teacher_width
    )
    return BertForNER(config)
