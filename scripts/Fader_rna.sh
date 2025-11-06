#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Multi-experiment Fader Network training launcher
# ============================================================

DATA_DIR="data/processed"
EXPNAME="Ribo"

LAT_DIS_DROPOUT=0.1
LAMBDA_LAT_DIS=1.0
BATCH_SIZE=64
EPOCHS=300

# ============================================================
# Architecture sweeps
# ============================================================
# Each index i corresponds across the three arrays
declare -a ENCODER_LIST=("50,20" "100,50,20" "200,100,50" "30,10")
declare -a DECODER_LIST=("50" "70,50" "100,50" "30")
declare -a DIS_LIST=("15,10,5" "30,20,10" "50,25,10" "10,5")

# Detect number of available CPUs (from PBS environment)
N_CPUS=${PBS_NCPUS:-$(nproc)}
echo "Detected $N_CPUS available CPUs."

# Limit parallel jobs to available CPUs
MAX_JOBS=$N_CPUS
running_jobs=0

# ============================================================
# Launch experiments in parallel
# ============================================================
for i in "${!ENCODER_LIST[@]}"; do
  ENCODER_DIMS="${ENCODER_LIST[$i]}"
  DECODER_DIMS="${DECODER_LIST[$i]}"
  DIS_HIDDEN_DIMS="${DIS_LIST[$i]}"

  OUTNAME="${EXPNAME}_enc${ENCODER_DIMS//,/x}_dec${DECODER_DIMS//,/x}_dis${DIS_HIDDEN_DIMS//,/x}"
  OUTDIR="logs/${OUTNAME}"
  mkdir -p "$OUTDIR"

  echo "[$(date)] Launching experiment: $OUTNAME"

  python -u train.py \
    --name "${OUTNAME}" \
    --data_path "${DATA_DIR}" \
    --seq_len 108 \
    --n_amino 5 \
    --alphabet_type rna \
    --attr binary \
    --encoder_hidden_dims "${ENCODER_DIMS}" \
    --decoder_hidden_dims "${DECODER_DIMS}" \
    --dis_hidden_dims "${DIS_HIDDEN_DIMS}" \
    --n_lat_dis 1 \
    --n_ptc_dis 0 \
    --n_clf_dis 0 \
    --lambda_ae 1.3 \
    --lambda_lat_dis ${LAMBDA_LAT_DIS} \
    --lat_dis_dropout ${LAT_DIS_DROPOUT} \
    --batch_size ${BATCH_SIZE} \
    --n_epochs ${EPOCHS} \
    --cuda False \
    > "${OUTDIR}/train.log" 2>&1 &

  ((running_jobs++))

  # Wait if we’ve hit the CPU limit
  if (( running_jobs >= MAX_JOBS )); then
    wait
    running_jobs=0
  fi
done

# Wait for any remaining jobs to finish
wait

echo "[$(date)] ✅ All experiments completed."
