# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# Adapted for 1D protein and RNA sequences by Jacopo Boccato & ChatGPT.

import os
import argparse
import torch

from src.loader import load_sequences, DataSampler
from src.utils import initialize_exp, bool_flag, attr_flag, check_attr
from src.model import AutoEncoder, LatentDiscriminator, PatchDiscriminator, Classifier
from src.training import Trainer
from src.evaluation import Evaluator


# ============================================================
# Parse parameters
# ============================================================
parser = argparse.ArgumentParser(description="Sequence autoencoder (dense Fader)")

# Experiment setup
parser.add_argument("--name", type=str, default="seq_autoencoder_exp")
parser.add_argument("--data_path", type=str, default="data")

# Sequence configuration
parser.add_argument("--seq_len", type=int, required=True,
                    help="Fixed sequence length (padding/truncation applied)")

parser.add_argument(
    "--alphabet_type",
    type=str,
    default="normal",
    choices=["normal", "lattice", "rna"],
    help="Alphabet used for one-hot encoding (normal / lattice / rna)"
)

parser.add_argument("--n_amino", type=int, default=None,
                    help="Number of alphabet symbols (auto-inferred if None)")

parser.add_argument("--attr", type=attr_flag, default="")

# Flexible dense architecture controls
parser.add_argument("--encoder_hidden_dims", type=str, default="",
                    help="Comma-separated hidden dimensions for encoder, e.g. '256,128'")
parser.add_argument("--decoder_hidden_dims", type=str, default="",
                    help="Comma-separated hidden dimensions for decoder BEFORE the final output dim, e.g. '130,256'")

# Latent + Discriminator
parser.add_argument("--hid_dim", type=int, default=512,
                    help="Latent dim (ignored if encoder_hidden_dims provided).")
parser.add_argument("--lat_dis_dropout", type=float, default=0.0)
parser.add_argument("--dis_hidden_dims", type=str, default="",
                    help="Comma-separated hidden dims for latent discriminator MLP (e.g., '128,64').")

# Discriminator toggles
parser.add_argument("--n_lat_dis", type=int, default=1)
parser.add_argument("--n_ptc_dis", type=int, default=0)
parser.add_argument("--n_clf_dis", type=int, default=0)

# ============================================================
# Loss weights
# ============================================================
parser.add_argument("--lambda_ae", type=float, default=1.0)
parser.add_argument("--lambda_lat_dis", type=float, default=1.0)
parser.add_argument("--lambda_ptc_dis", type=float, default=0.0)
parser.add_argument("--lambda_clf_dis", type=float, default=0.0)
parser.add_argument("--lambda_schedule", type=float, default=500000)

# ============================================================
# Optimization
# ============================================================
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--ae_optimizer", type=str, default="adam,lr=0.0002")
parser.add_argument("--dis_optimizer", type=str, default="adam,lr=0.0002")
parser.add_argument("--clip_grad_norm", type=float, default=5.0)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--epoch_size", type=int, default=50000)

# ============================================================
# Reload / evaluation / misc
# ============================================================
parser.add_argument("--ae_reload", type=str, default="")
parser.add_argument("--lat_dis_reload", type=str, default="")
parser.add_argument("--ptc_dis_reload", type=str, default="")
parser.add_argument("--clf_dis_reload", type=str, default="")
parser.add_argument("--eval_clf", type=str, default="")
parser.add_argument("--debug", type=bool_flag, default=False)
parser.add_argument("--cuda", type=bool_flag, default=True)

params = parser.parse_args()


# ============================================================
# Device setup
# ============================================================
if not torch.cuda.is_available():
    params.cuda = False
device = torch.device("cuda" if params.cuda else "cpu")

# ============================================================
# Parse layer dimensions
# ============================================================
def _parse_dims(s: str):
    s = (s or "").strip()
    return [int(x) for x in s.split(",") if x] if s else []

params.encoder_hidden_dims = _parse_dims(params.encoder_hidden_dims)
params.decoder_hidden_dims = _parse_dims(params.decoder_hidden_dims)
params.dis_hidden_dims = _parse_dims(params.dis_hidden_dims)

# ============================================================
# Sanity checks and alphabet-dependent defaults
# ============================================================
check_attr(params)
assert len(params.name.strip()) > 0, "Experiment name cannot be empty."

if params.n_amino is None:
    if params.alphabet_type == "normal":
        params.n_amino = 21
    elif params.alphabet_type == "lattice":
        params.n_amino = 20
    elif params.alphabet_type == "rna":
        params.n_amino = 5  # A,C,G,U, and '-'
    else:
        raise ValueError(f"Invalid alphabet_type: {params.alphabet_type}")

assert (params.lambda_lat_dis == 0.0) or (params.n_lat_dis > 0)
assert (params.lambda_ptc_dis == 0.0) or (params.n_ptc_dis > 0)
assert (params.lambda_clf_dis == 0.0) or (params.n_clf_dis > 0)

if not params.attr:
    params.attr = []

# ============================================================
# Initialize experiment and dataset
# ============================================================
logger = initialize_exp(params)

logger.info(f"Loading dataset using '{params.alphabet_type}' alphabet ({params.n_amino} symbols)...")
data, labels = load_sequences(params, alphabet_type=params.alphabet_type)
train_data = DataSampler(data[0], labels[0], params)
valid_data = DataSampler(data[1], labels[1], params)
logger.info(f"Dataset loaded: {len(train_data)} train / {len(valid_data)} valid samples")

# ============================================================
# Build models
# ============================================================
ae = AutoEncoder(params)
lat_dis = LatentDiscriminator(params) if params.n_lat_dis > 0 else None
ptc_dis = PatchDiscriminator(params) if params.n_ptc_dis > 0 else None
clf_dis = Classifier(params) if params.n_clf_dis > 0 else None

# Move models to device
ae = ae.to(device)
if lat_dis is not None:
    lat_dis = lat_dis.to(device)
if ptc_dis is not None:
    ptc_dis = ptc_dis.to(device)
if clf_dis is not None:
    clf_dis = clf_dis.to(device)

# ============================================================
# Trainer and Evaluator
# ============================================================
trainer = Trainer(ae, lat_dis, ptc_dis, clf_dis, train_data, params)
# eval_clf is unused in sequence version; pass None for interface compatibility
evaluator = Evaluator(ae, lat_dis, ptc_dis, clf_dis, None, valid_data, params)

# ============================================================
# Training loop
# ============================================================
for n_epoch in range(params.n_epochs):
    logger.info(f"Starting epoch {n_epoch}...")

    # Iterate "epoch_size" examples in chunks of batch_size
    for n_iter in range(0, params.epoch_size, params.batch_size):

        # 1) Latent discriminator(s)
        for _ in range(params.n_lat_dis):
            trainer.lat_dis_step()

        # 2) Patch discriminator(s) (disabled by default for dense AE)
        for _ in range(params.n_ptc_dis):
            trainer.ptc_dis_step()

        # 3) Classifier discriminator(s) (off by default)
        for _ in range(params.n_clf_dis):
            trainer.clf_dis_step()

        # 4) Autoencoder (reconstruction + adversarial terms)
        trainer.autoencoder_step()

        # Logging every ~25 steps (handled inside .step)
        trainer.step(n_iter)

    # ===== End of epoch: evaluate & checkpoint =====
    to_log = evaluator.evaluate(n_epoch)
    trainer.save_best_periodic(to_log)
    logger.info(f"End of epoch {n_epoch}.\n")
