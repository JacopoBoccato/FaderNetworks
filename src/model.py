# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# Adapted for 1D protein sequences by Jacopo Boccato & ChatGPT.

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

def build_dense_layers(input_dim, hidden_dims):
    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(nn.ReLU())
        prev_dim = h
    if layers:
        layers = layers[:-1]  # remove last ReLU (optional)
    return nn.Sequential(*layers)


class AutoEncoder(nn.Module):
    def __init__(self, params):
        super(AutoEncoder, self).__init__()

        self.seq_len = params.seq_len
        self.n_amino = params.n_amino
        self.input_dim = self.seq_len * self.n_amino
        self.attr_dim = getattr(params, "n_attr", 0)

        # Encoder: input_dim → encoder_hidden_dims
        self.encoder = build_dense_layers(self.input_dim, params.encoder_hidden_dims)
        self.latent_dim = params.encoder_hidden_dims[-1]

        # Decoder: latent_dim+attr_dim → decoder_hidden_dims → input_dim
        self.decoder = build_dense_layers(self.latent_dim + self.attr_dim,
                                          params.decoder_hidden_dims + [self.input_dim])

        params.hid_dim = self.latent_dim

    def encode(self, x):
        bs = x.size(0)
        x_flat = x.view(bs, -1)
        z = self.encoder(x_flat)
        return [None] * (len(self.encoder) // 2 - 1) + [z]

    def decode(self, enc_outputs, y=None):
        z = enc_outputs[-1]
        if y is not None:
            z = torch.cat([z, y], dim=1)
        x_hat = self.decoder(z)
        x_hat = x_hat.view(-1, self.n_amino, self.seq_len)
        return [None] * (len(self.decoder) // 2 - 1) + [x_hat]

    def forward(self, x, y=None):
        enc_outputs = self.encode(x)
        dec_outputs = self.decode(enc_outputs, y)
        return enc_outputs, dec_outputs

class LatentDiscriminator(nn.Module):
    def __init__(self, params):
        super(LatentDiscriminator, self).__init__()
        self.n_attr = getattr(params, "n_attr", 0)
        in_dim = getattr(params, "hid_dim", None)
        assert in_dim is not None

        dims = [in_dim] + params.dis_hidden_dims + [self.n_attr or 1]
        self.net = build_dense_layers(dims[0], dims[1:])

    def forward(self, z):
        if z.dim() == 3:
            z = z.mean(dim=2)
        return self.net(z)

class PatchDiscriminator(nn.Module):
    """
    Dense (MLP) discriminator for sequences.
    Input: x of shape (B, n_amino, seq_len)
    Output: probability in (0,1) per sequence (B,)
    """
    def __init__(self, params, hidden_dims=None, dropout=0.0):
        super(PatchDiscriminator, self).__init__()
        self.seq_len = params.seq_len
        self.n_amino = params.n_amino

        in_dim = self.seq_len * self.n_amino
        hidden_dims = hidden_dims or [512, 256]

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2, inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]  # keep Sigmoid to match BCE(prob) in Trainer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, n_amino, seq_len)
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        out = self.net(x)           # (B, 1)
        return out.squeeze(1)       # (B,)


class Classifier(nn.Module):
    def __init__(self, params):
        super(Classifier, self).__init__()
        assert getattr(params, "n_attr", 0) > 0

        in_dim = getattr(params, "hid_dim", None)
        dims = [in_dim] + params.dis_hidden_dims + [params.n_attr]
        self.net = build_dense_layers(dims[0], dims[1:])

    def forward(self, z):
        if z.dim() == 3:
            z = z.mean(dim=2)
        return self.net(z)

def _parse_dims(s):
    return [int(x) for x in s.split(",") if x]

def get_attr_loss(output, attributes, flip, params):
    assert isinstance(flip, bool)
    k = 0
    loss = 0
    for (_, n_cat) in params.attr:
        x = output[:, k:k + n_cat].contiguous()
        y = attributes[:, k:k + n_cat].max(1)[1].view(-1)
        if flip:
            shift = torch.LongTensor(y.size()).random_(n_cat - 1) + 1
            if x.is_cuda:
                shift = shift.cuda()
            y = (y + Variable(shift)) % n_cat
        loss += F.cross_entropy(x, y)
        k += n_cat
    return loss

def sequence_cross_entropy(x_hat, x_true):
    """
    Cross-entropy for sequence reconstruction.
    x_hat: (B, V, T) raw logits
    x_true: (B, V, T) one-hot ground truth
    """
    B, V, T = x_hat.shape

    # targets: (B*T,)
    target = x_true.argmax(dim=1).reshape(B * T)

    # logits: (B*T, V)
    logits = x_hat.permute(0, 2, 1).contiguous().reshape(B * T, V)

    return F.cross_entropy(logits, target, reduction='mean')


def update_predictions(all_preds, preds, targets, params):
    assert len(all_preds) == len(params.attr)
    k = 0
    for j, (_, n_cat) in enumerate(params.attr):
        _preds = preds[:, k:k + n_cat].max(1)[1]
        _targets = targets[:, k:k + n_cat].max(1)[1]
        all_preds[j].extend((_preds == _targets).tolist())
        k += n_cat
    assert k == params.n_attr

def get_mappings(params):
    if not hasattr(params, "mappings"):
        mappings = []
        k = 0
        for (_, n_cat) in params.attr:
            mappings.append((k, k + n_cat))
            k += n_cat
        assert k == params.n_attr
        params.mappings = mappings
    return params.mappings

def flip_attributes(attributes, params, attribute_id, new_value=None):
    assert attributes.size(1) == params.n_attr
    mappings = get_mappings(params)
    attributes = attributes.data.clone().cpu()

    def flip_attribute(attribute_id, new_value=None):
        bs = attributes.size(0)
        i, j = mappings[attribute_id]
        attributes[:, i:j].zero_()
        if new_value is None:
            y = torch.LongTensor(bs).random_(j - i)
        else:
            y = torch.LongTensor(bs).fill_(new_value)
        attributes[:, i:j].scatter_(1, y.unsqueeze(1), 1)

    if attribute_id == "all":
        for aid in range(len(params.attr)):
            flip_attribute(aid)
    else:
        flip_attribute(attribute_id, new_value)

    return Variable(attributes.cuda() if attributes.is_cuda else attributes)
