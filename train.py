# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# Adapted for protein sequences by [Your Name]
#

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
parser = argparse.ArgumentParser(description="Protein sequence autoencoder")

# Experiment setup
parser.add_argument("--name", type=str, default="protein_seq_exp")
parser.add_argument("--data_path", type=str, default="data")

# Protein sequence configuration
parser.add_argument("--seq_len", type=int, required=True,
                    help="Fixed sequence length (padding/truncation applied)")
parser.add_argument("--n_amino", type=int, default=21,
                    help="Number of amino acid types (channels)")
parser.add_argument("--alphabet_type", type=str, default="normal",
                    choices=["normal", "lattice"],
                    help="Alphabet used for one-hot encoding")
parser.add_argument("--attr", type=attr_flag, default="Attribute1,Attribute2")

# Architecture
parser.add_argument("--instance_norm", type=bool_flag, default=False)
parser.add_argument("--init_fm", type=int, default=32)
parser.add_argument("--max_fm", type=int, default=512)
parser.add_argument("--n_layers", type=int, default=6)
parser.add_argument("--n_skip", type=int, default=0)
parser.add_argument("--deconv_method", type=str, default="convtranspose",
                    choices=["convtranspose", "upsampling", "pixelshuffle"])
parser.add_argument("--hid_dim", type=int, default=512)
parser.add_argument("--dec_dropout", type=float, default=0.0)
parser.add_argument("--lat_dis_dropout", type=float, default=0.3)

# Discriminator settings
parser.add_argument("--n_lat_dis", type=int, default=1)
parser.add_argument("--n_ptc_dis", type=int, default=0)
parser.add_argument("--n_clf_dis", type=int, default=0)

# Loss weights
parser.add_argument("--lambda_ae", type=float, default=1.0)
parser.add_argument("--lambda_lat_dis", type=float, default=0.0001)
parser.add_argument("--lambda_ptc_dis", type=float, default=0.0)
parser.add_argument("--lambda_clf_dis", type=float, default=0.0)
parser.add_argument("--lambda_schedule", type=float, default=500000)

# Optimization
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--ae_optimizer", type=str, default="adam,lr=0.0002")
parser.add_argument("--dis_optimizer", type=str, default="adam,lr=0.0002")
parser.add_argument("--clip_grad_norm", type=float, default=5.0)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--epoch_size", type=int, default=50000)

# Reload options
parser.add_argument("--ae_reload", type=str, default="")
parser.add_argument("--lat_dis_reload", type=str, default="")
parser.add_argument("--ptc_dis_reload", type=str, default="")
parser.add_argument("--clf_dis_reload", type=str, default="")

# Evaluation
parser.add_argument("--eval_clf", type=str, default="")

# Misc
parser.add_argument("--debug", type=bool_flag, default=False)
parser.add_argument("--cuda", type=bool_flag, default=True)

params = parser.parse_args()

# ============================================================
# Sanity checks
# ============================================================
check_attr(params)
assert len(params.name.strip()) > 0
assert params.n_skip <= params.n_layers - 1
assert params.lambda_lat_dis == 0 or params.n_lat_dis > 0

# ============================================================
# Initialize experiment and dataset
# ============================================================
logger = initialize_exp(params)
logger.info(f"Loading dataset using '{params.alphabet_type}' alphabet ...")
data, labels = load_sequences(params, alphabet_type=params.alphabet_type)
train_data = DataSampler(data[0], labels[0], params)
valid_data = DataSampler(data[1], labels[1], params)
logger.info(f"Dataset loaded: {len(train_data)} train / {len(valid_data)} valid samples")

# ============================================================
# Build models
# ============================================================
ae = AutoEncoder(params).cuda() if params.cuda else AutoEncoder(params)
lat_dis = LatentDiscriminator(params).cuda() if params.n_lat_dis and params.cuda else None
ptc_dis = PatchDiscriminator(params).cuda() if params.n_ptc_dis and params.cuda else None
clf_dis = Classifier(params).cuda() if params.n_clf_dis and params.cuda else None

eval_clf = None
if params.eval_clf and os.path.isfile(params.eval_clf):
    eval_clf = torch.load(params.eval_clf)
    if params.cuda:
        eval_clf = eval_clf.cuda()
    eval_clf.eval()

# ============================================================
# Trainer & Evaluator
# ============================================================
trainer = Trainer(ae, lat_dis, ptc_dis, clf_dis, train_data, params)
evaluator = Evaluator(ae, lat_dis, ptc_dis, clf_dis, eval_clf, valid_data, params)

# ============================================================
# Training loop
# ============================================================
for n_epoch in range(params.n_epochs):

    logger.info(f"Starting epoch {n_epoch}...")

    for n_iter in range(0, params.epoch_size, params.batch_size):

        for _ in range(params.n_lat_dis):
            trainer.lat_dis_step()

        for _ in range(params.n_ptc_dis):
            trainer.ptc_dis_step()

        for _ in range(params.n_clf_dis):
            trainer.clf_dis_step()

        trainer.autoencoder_step()
        trainer.step(n_iter)

    # Evaluate & save models
    to_log = evaluator.evaluate(n_epoch)
    trainer.save_best_periodic(to_log)
    logger.info(f"End of epoch {n_epoch}.\n")
