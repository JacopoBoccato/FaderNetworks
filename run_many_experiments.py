#!/usr/bin/env python3
"""Create dummy dataset and run multiple train sessions (binary/continuous)."""

import os
import random
import subprocess
import argparse
from pathlib import Path

import torch

from src.loader import onehot_encode_sequences, get_alphabet


def make_random_sequences(n_samples, seq_len, alphabet_type, pad_token='-'):
    alphabet = get_alphabet(alphabet_type)
    alphabet_no_pad = [a for a in alphabet if a != pad_token]
    sequences = []
    for _ in range(n_samples):
        seq = ''.join(random.choices(alphabet_no_pad, k=seq_len))
        rand_pos = random.randint(0, seq_len - 1)
        seq = seq[:rand_pos] + pad_token + seq[rand_pos + 1:]
        sequences.append(seq)
    return sequences


def save_dummy_dataset(data_path, seq_len, n_train, n_valid, alphabet_type, x_type='onehot'):
    os.makedirs(data_path, exist_ok=True)

    if x_type == 'onehot':
        train_seq = make_random_sequences(n_train, seq_len, alphabet_type)
        valid_seq = make_random_sequences(n_valid, seq_len, alphabet_type)
        train_X, _ = onehot_encode_sequences(train_seq, alphabet_type, seq_len)
        valid_X, _ = onehot_encode_sequences(valid_seq, alphabet_type, seq_len)
    elif x_type == 'indices':
        n_symbols = len(get_alphabet(alphabet_type))
        train_X = torch.randint(0, n_symbols, (n_train, seq_len), dtype=torch.long)
        valid_X = torch.randint(0, n_symbols, (n_valid, seq_len), dtype=torch.long)
    else:
        raise ValueError(f"Unknown x_type: {x_type}")

    # Binary label dummy data for training
    train_y = torch.randint(0, 2, (n_train, 2), dtype=torch.float32)
    valid_y = torch.randint(0, 2, (n_valid, 2), dtype=torch.float32)

    torch.save(train_X, os.path.join(data_path, f"sequences_{seq_len}.pth"))
    torch.save(train_y, os.path.join(data_path, f"labels_{seq_len}.pth"))

    # optional valid collections
    torch.save(valid_X, os.path.join(data_path, f"valid_sequences_{seq_len}.pth"))
    torch.save(valid_y, os.path.join(data_path, f"valid_labels_{seq_len}.pth"))

    print(f"Saved dummy dataset to {data_path} with seq_len={seq_len}, train={n_train}, valid={n_valid} (x_type={x_type})")


def run_training(data_path, seq_len, label_type, experiment_name, epochs, batch_size, x_type='onehot', extra_args=None):
    label_str = label_type
    if label_type == 'binary':
        attr_arg = '--attr BinaryAttr'
    else:
        attr_arg = ''
    cmd = (
        f"python train.py --seq_len {seq_len} --data_path {data_path} --name {experiment_name} "
        f"--x_type {x_type} "
        f"--label_type {label_str} --n_epochs {epochs} --batch_size {batch_size} "
        f"--lambda_lat_dis 1.0 --lambda_ptc_dis 0.0 --lambda_clf_dis 0.0 --n_lat_dis 1 "
        f"{attr_arg} {extra_args or ''}"
    ).strip()

    print(f"Running: {cmd}")
    subprocess.run(["bash", "-lc", cmd], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dummy set and run N trainings.')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_valid', type=int, default=100)
    parser.add_argument('--alphabet_type', type=str, default='normal')
    parser.add_argument('--n_runs', type=int, default=3)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--label_types', type=str, default='binary,continuous')
    parser.add_argument('--x_type', type=str, default='onehot', choices=['onehot', 'indices'])
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Make sure models directory exists for train.py checkpoints
    models_dir = Path(__file__).resolve().parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    save_dummy_dataset(args.data_path, args.seq_len, args.n_train, args.n_valid, args.alphabet_type, x_type=args.x_type)

    label_types = [s.strip() for s in args.label_types.split(',') if s.strip()]

    for label_type in label_types:
        for run_id in range(1, args.n_runs + 1):
            name = f"exp_{label_type}_run{run_id:02d}_{args.x_type}"
            run_training(
                args.data_path,
                args.seq_len,
                label_type,
                name,
                args.n_epochs,
                args.batch_size,
                x_type=args.x_type,
                extra_args="--encoder_hidden_dims 1024,256 --decoder_hidden_dims 256,1024 --dis_hidden_dims 128,64"
            )

    print('✅ All experiments completed.')
