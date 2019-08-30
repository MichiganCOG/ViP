# Originally from https://github.com/salesforce/densecap
"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
# Last modified by Luowei Zhou on 07/01/2018

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import random
import string
import sys
import math
import uuid
import numpy as np

INF = 1e10

def positional_encodings_like(x, t=None):
    if t is None:
        positions = torch.arange(0, x.size(1))
        if x.is_cuda:
           positions = positions.cuda(x.get_device())
    else:
        positions = t
    encodings = x.new(*x.size()[1:]).fill_(0)
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())


    for channel in range(x.size(-1)):
        if channel % 2 == 0:
            encodings[:, channel] = torch.sin(
                positions.float() / 10000 ** (channel / x.size(2)))
        else:
            encodings[:, channel] = torch.cos(
                positions.float() / 10000 ** ((channel - 1) / x.size(2)))
    return Variable(encodings)

class Linear(nn.Linear):

    def forward(self, x):
        size = x.size()
        return super().forward(
            x.contiguous().view(-1, size[-1])).view(*size[:-1], -1)

# F.softmax has strange default behavior, normalizing over dim 0 for 3D inputs
# deprecated since PyTorch 0.3
# def softmax(x):
#     if x.dim() == 3:
#         return F.softmax(x.transpose(0, 2)).transpose(0, 2)
#     return F.softmax(x)

# torch.matmul can't do (4, 3, 2) @ (4, 2) -> (4, 3)
def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-2)).squeeze(-2)

class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, drop_ratio):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)

    def forward(self, *x):
        return self.layernorm(x[0] + self.dropout(self.layer(*x)))

class Attention(nn.Module):

    def __init__(self, d_key, drop_ratio, causal):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal

    def forward(self, query, key, value):
        dot_products = matmul(query, key.transpose(1, 2))
        if query.dim() == 3 and (self is None or self.causal):
            tri = torch.ones(key.size(1), key.size(1)).triu(1) * INF
            if key.is_cuda:
                tri = tri.cuda(key.get_device())
            dot_products.data.sub_(tri.unsqueeze(0))
        return matmul(self.dropout(F.softmax(dot_products / self.scale, dim=2)), value)

class MultiHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, drop_ratio, causal=False):
        super().__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal)
        self.wq = Linear(d_key, d_key, bias=False)
        self.wk = Linear(d_key, d_key, bias=False)
        self.wv = Linear(d_value, d_value, bias=False)
        self.wo = Linear(d_value, d_key, bias=False)
        self.n_heads = n_heads

    def forward(self, query, key, value):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (
            x.chunk(self.n_heads, -1) for x in (query, key, value))
        return self.wo(torch.cat([self.attention(q, k, v)
                          for q, k, v in zip(query, key, value)], -1))

class FeedForward(nn.Module):

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.linear1 = Linear(d_model, d_hidden)
        self.linear2 = Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_hidden, n_heads, drop_ratio):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio),
            d_model, drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)

    def forward(self, x):
        return self.feedforward(self.selfattn(x, x, x))

class Encoder(nn.Module):

    def __init__(self, d_model, d_hidden, n_vocab, n_layers, n_heads,
                 drop_ratio):
        super().__init__()
        # self.linear = nn.Linear(d_model*2, d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_hidden, n_heads, drop_ratio)
             for i in range(n_layers)])
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x, mask=None):
        # x = self.linear(x)
        x = x+positional_encodings_like(x)
        x = self.dropout(x)
        if mask is not None:
            x = x*mask
        encoding = []
        for layer in self.layers:
            x = layer(x)
            if mask is not None:
                x = x*mask
            encoding.append(x)
        return encoding

class Transformer(nn.Module):

    def __init__(self, d_model, n_vocab_src, vocab_trg, d_hidden=2048,
                 n_layers=6, n_heads=8, drop_ratio=0.1):
        super().__init__()
        self.encoder = Encoder(d_model, d_hidden, n_vocab_src, n_layers,
                               n_heads, drop_ratio)

    def denum(self, data):
        return ' '.join(self.decoder.vocab.itos[i] for i in data).replace(
            ' <eos>', '#').replace(' <pad>', '')

    def forward(self, x):
        encoding = self.encoder(x)

        return encoding[-1], encoding

