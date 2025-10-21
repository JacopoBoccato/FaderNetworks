# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# Adapted for 1D protein sequences by Jacopo Boccato & ChatGPT.
#

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


def build_layers(seq_len, n_amino, init_fm, max_fm, n_layers, n_attr, n_skip,
                 deconv_method, instance_norm, enc_dropout, dec_dropout):
    """
    Build 1D auto-encoder layers for protein sequences.
    Each encoder layer halves the sequence length, and each decoder layer upsamples it.
    """
    assert init_fm <= max_fm
    assert n_skip <= n_layers - 1
    assert np.log2(seq_len).is_integer(), "seq_len must be a power of 2"
    assert n_layers <= int(np.log2(seq_len))
    assert isinstance(instance_norm, bool)
    assert 0 <= enc_dropout < 1
    assert 0 <= dec_dropout < 1

    norm_fn = nn.InstanceNorm1d if instance_norm else nn.BatchNorm1d
    enc_layers, dec_layers = [], []

    n_in = n_amino
    n_out = init_fm

    for i in range(n_layers):
        enc_layer, dec_layer = [], []
        skip_connection = n_layers - (n_skip + 1) <= i < n_layers - 1
        n_dec_in = n_out + n_attr + (n_out if skip_connection else 0)
        n_dec_out = n_in

        # encoder layer
        enc_layer.append(nn.Conv1d(n_in, n_out, 4, 2, 1))
        if i > 0:
            enc_layer.append(norm_fn(n_out, affine=True))
        enc_layer.append(nn.LeakyReLU(0.2, inplace=True))
        if enc_dropout > 0:
            enc_layer.append(nn.Dropout(enc_dropout))

        # decoder layer
        if deconv_method == "upsampling":
            dec_layer.append(nn.Upsample(scale_factor=2, mode="nearest"))
            dec_layer.append(nn.Conv1d(n_dec_in, n_dec_out, 3, 1, 1))
        elif deconv_method == "convtranspose":
            dec_layer.append(nn.ConvTranspose1d(n_dec_in, n_dec_out, 4, 2, 1, bias=False))
        else:
            raise ValueError("Unsupported deconv_method for 1D data")

        if i > 0:
            dec_layer.append(norm_fn(n_dec_out, affine=True))
            if dec_dropout > 0 and i >= n_layers - 3:
                dec_layer.append(nn.Dropout(dec_dropout))
            dec_layer.append(nn.ReLU(inplace=True))
        else:
            dec_layer.append(nn.Tanh())

        # update
        n_in = n_out
        n_out = min(2 * n_out, max_fm)
        enc_layers.append(nn.Sequential(*enc_layer))
        dec_layers.insert(0, nn.Sequential(*dec_layer))

    return enc_layers, dec_layers


class AutoEncoder(nn.Module):
    """
    1D Autoencoder for protein sequences.
    Input:  [B, n_amino, seq_len]
    Output: [B, n_amino, seq_len]
    """

    def __init__(self, params):
        super(AutoEncoder, self).__init__()

        self.seq_len = params.seq_len
        self.n_amino = params.n_amino
        self.instance_norm = params.instance_norm
        self.init_fm = params.init_fm
        self.max_fm = params.max_fm
        self.n_layers = params.n_layers
        self.n_skip = params.n_skip
        self.deconv_method = params.deconv_method
        self.dropout = params.dec_dropout
        self.attr = getattr(params, "attr", [])
        self.n_attr = getattr(params, "n_attr", 0)

        enc_layers, dec_layers = build_layers(
            seq_len=self.seq_len,
            n_amino=self.n_amino,
            init_fm=self.init_fm,
            max_fm=self.max_fm,
            n_layers=self.n_layers,
            n_attr=self.n_attr,
            n_skip=self.n_skip,
            deconv_method=self.deconv_method,
            instance_norm=self.instance_norm,
            enc_dropout=0,
            dec_dropout=self.dropout
        )
        self.enc_layers = nn.ModuleList(enc_layers)
        self.dec_layers = nn.ModuleList(dec_layers)

    def encode(self, x):
        assert x.dim() == 3 and x.size(1) == self.n_amino and x.size(2) == self.seq_len, \
            f"Expected [B,{self.n_amino},{self.seq_len}], got {tuple(x.shape)}"
        enc_outputs = [x]
        for layer in self.enc_layers:
            enc_outputs.append(layer(enc_outputs[-1]))
        return enc_outputs

    def decode(self, enc_outputs, y=None):
        bs = enc_outputs[0].size(0)
        if self.n_attr == 0:
            y = torch.empty(bs, 0, device=enc_outputs[0].device)
        else:
            assert y is not None and y.size() == (bs, self.n_attr)

        dec_outputs = [enc_outputs[-1]]
        for i, layer in enumerate(self.dec_layers):
            length = dec_outputs[-1].size(2)
            inputs = [dec_outputs[-1]]

            if self.n_attr > 0:
                y_exp = y.unsqueeze(2).expand(bs, self.n_attr, length)
                inputs.append(y_exp)

            if 0 < i <= self.n_skip:
                inputs.append(enc_outputs[-1 - i])

            inputs = torch.cat(inputs, 1)
            dec_outputs.append(layer(inputs))

        assert dec_outputs[-1].size(1) == self.n_amino
        return dec_outputs

    def forward(self, x, y=None):
        enc_outputs = self.encode(x)
        dec_outputs = self.decode(enc_outputs, y)
        return enc_outputs, dec_outputs


class LatentDiscriminator(nn.Module):
    def __init__(self, params):
        super(LatentDiscriminator, self).__init__()

        self.seq_len = params.seq_len
        self.n_amino = params.n_amino
        self.init_fm = params.init_fm
        self.max_fm = params.max_fm
        self.n_layers = params.n_layers
        self.n_skip = params.n_skip
        self.hid_dim = params.hid_dim
        self.dropout = params.lat_dis_dropout
        self.attr = getattr(params, "attr", [])
        self.n_attr = getattr(params, "n_attr", 0)

        # how deep we go overall (from full-resolution input)
        self.n_dis_layers = int(np.log2(self.seq_len))

        # shape the discriminator will receive from the encoder
        self.conv_in_len = self.seq_len // (2 ** (self.n_layers - self.n_skip))
        self.conv_in_fm  = min(self.init_fm * (2 ** (self.n_layers - self.n_skip - 1)), self.max_fm)
        self.conv_out_fm = min(self.init_fm * (2 ** (self.n_dis_layers - 1)), self.max_fm)

        # Build encoder-like layers from the full input scale,
        # then reuse ONLY the suffix starting at the same depth as the bottleneck.
        enc_layers, _ = build_layers(
            seq_len=self.seq_len,       # full scale (not conv_in_len)
            n_amino=self.n_amino,       # from input channels (21)
            init_fm=self.init_fm,
            max_fm=self.max_fm,
            n_layers=self.n_dis_layers, # full depth
            n_attr=self.n_attr,
            n_skip=0,
            deconv_method='convtranspose',
            instance_norm=False,
            enc_dropout=self.dropout,
            dec_dropout=0.0
        )

        # *** Key line: take only the suffix so the first reused layer
        # expects 'conv_in_fm' channels, matching the encoder bottleneck. ***
        self.conv_layers = nn.Sequential(*(enc_layers[self.n_layers - self.n_skip:]))

        self.proj_layers = nn.Sequential(
            nn.Linear(self.conv_out_fm, self.hid_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hid_dim, self.n_attr)
        )

    def forward(self, x):
        # Expect the encoder feature at the bottleneck depth:
        # x shape: [B, conv_in_fm, conv_in_len] e.g. [B, 512, 2]
        assert x.dim() == 3 and x.size(1) == self.conv_in_fm and x.size(2) == self.conv_in_len, \
            f"Expected [B,{self.conv_in_fm},{self.conv_in_len}], got {tuple(x.shape)}"
        conv_output = self.conv_layers(x)
        # collapse length by global average pooling (1D)
        conv_output = conv_output.mean(dim=2)
        return self.proj_layers(conv_output)


class PatchDiscriminator(nn.Module):
    """1D PatchGAN discriminator (for optional adversarial reconstruction)."""

    def __init__(self, params):
        super(PatchDiscriminator, self).__init__()
        self.seq_len = params.seq_len
        self.n_amino = params.n_amino
        self.init_fm = params.init_fm
        self.max_fm = params.max_fm

        layers = [
            nn.Conv1d(self.n_amino, self.init_fm, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        n_in = self.init_fm
        n_out = min(2 * n_in, self.max_fm)

        for _ in range(3):
            layers += [
                nn.Conv1d(n_in, n_out, 4, 2, 1),
                nn.BatchNorm1d(n_out),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            n_in = n_out
            n_out = min(2 * n_out, self.max_fm)

        layers += [
            nn.Conv1d(n_out, 1, 4, 1, 1),
            nn.Sigmoid()
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 3
        out = self.layers(x)
        return out.mean(dim=[1, 2])  # mean over sequence


class Classifier(nn.Module):
    """Optional classifier for supervised evaluation of attributes."""

    def __init__(self, params):
        super(Classifier, self).__init__()
        assert getattr(params, "n_attr", 0) > 0, "Classifier requires n_attr > 0"

        self.seq_len = params.seq_len
        self.n_amino = params.n_amino
        self.init_fm = params.init_fm
        self.max_fm = params.max_fm
        self.hid_dim = params.hid_dim
        self.n_attr = params.n_attr

        self.n_clf_layers = int(np.log2(self.seq_len))
        self.conv_out_fm = min(self.init_fm * (2 ** (self.n_clf_layers - 1)), self.max_fm)

        enc_layers, _ = build_layers(
            seq_len=self.seq_len,
            n_amino=self.n_amino,
            init_fm=self.init_fm,
            max_fm=self.max_fm,
            n_layers=self.n_clf_layers,
            n_attr=self.n_attr,
            n_skip=0,
            deconv_method="convtranspose",
            instance_norm=False,
            enc_dropout=0,
            dec_dropout=0
        )

        self.conv_layers = nn.Sequential(*enc_layers)
        self.proj_layers = nn.Sequential(
            nn.Linear(self.conv_out_fm, self.hid_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hid_dim, self.n_attr)
        )

    def forward(self, x):
        conv_output = self.conv_layers(x)
        conv_output = conv_output.mean(dim=2)
        return self.proj_layers(conv_output)


# ===========================================================
# Helper functions (unchanged from original, work as-is)
# ===========================================================

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
            assert n_cat >= 2
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
            assert new_value in range(j - i)
            y = torch.LongTensor(bs).fill_(new_value)
        attributes[:, i:j].scatter_(1, y.unsqueeze(1), 1)

    if attribute_id == "all":
        for attribute_id in range(len(params.attr)):
            flip_attribute(attribute_id)
    else:
        assert isinstance(new_value, int)
        flip_attribute(attribute_id, new_value)

    return Variable(attributes.cuda() if attributes.is_cuda else attributes)
