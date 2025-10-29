#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Minimal dense Fader Network training (AE + latent discriminator only)
# ============================================================

# --- Configuration ---
DATA_DIR="data/lattice/processed"
EXPNAME="lattice_dense_minimal"

# --- Model architecture ---
ENCODER_DIMS="120,60"
DECODER_DIMS="120"

# --- Discriminator and training ---
DIS_HIDDEN_DIMS="50,10"        # empty = simple perceptron
LAT_DIS_DROPOUT=0.2
LAMBDA_LAT_DIS=1.0
BATCH_SIZE=64
EPOCHS=500

# ============================================================
# Run training
# ============================================================
python -u train.py \
  --name "${EXPNAME}" \
  --data_path "${DATA_DIR}" \
  --seq_len 27 \
  --n_amino 20 \
  --alphabet_type lattice \
  --attr binary \
  --encoder_hidden_dims "${ENCODER_DIMS}" \
  --decoder_hidden_dims "${DECODER_DIMS}" \
  --dis_hidden_dims "${DIS_HIDDEN_DIMS}" \
  --n_lat_dis 1 \
  --n_ptc_dis 0 \
  --n_clf_dis 0 \
  --lambda_ae 3.0 \
  --lambda_lat_dis ${LAMBDA_LAT_DIS} \
  --lat_dis_dropout ${LAT_DIS_DROPOUT} \
  --batch_size ${BATCH_SIZE} \
  --n_epochs ${EPOCHS} \
  --cuda False
