#!/usr/bin/env python3
import os
from pathlib import Path
import torch
import argparse

# Add root directory to import src/loader
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.loader import onehot_encode_sequences

ALPHABET_TYPE = "lattice"

def read_sequences(path: Path):
    """Reads alternating score/sequence lines and returns list of sequences."""
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) % 2 != 0:
        raise ValueError(f"{path} has an odd number of lines (expected score/sequence pairs).")
    return [lines[i + 1].upper() for i in range(0, len(lines), 2)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fileA", type=str, required=True, help="Path to LatticeA.txt")
    parser.add_argument("--fileB", type=str, required=True, help="Path to LatticeB.txt")
    parser.add_argument("--outdir", type=str, required=True, help="Where to write .pth files")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seqs_a = read_sequences(Path(args.fileA))
    seqs_b = read_sequences(Path(args.fileB))

    all_seqs = seqs_a + seqs_b
    print(f"[INFO] Read {len(seqs_a)} A + {len(seqs_b)} B sequences.")

    # One-hot encode
    seq_tensor, alphabet = onehot_encode_sequences(
        all_seqs,
        alphabet_type=ALPHABET_TYPE,
        seq_len=None  # Will auto-compute max length
    )
    print(f"[INFO] One-hot shape: {seq_tensor.shape}")

    # Labels: A → [1, 0], B → [0, 1]
    attr_a = torch.zeros((len(seqs_a), 2)); attr_a[:, 0] = 1.0
    attr_b = torch.zeros((len(seqs_b), 2)); attr_b[:, 1] = 1.0
    label_tensor = torch.cat([attr_a, attr_b], dim=0)

    # Save as .pth
    seq_out = outdir / f"sequences_{seq_tensor.shape[-1]}.pth"
    label_out = outdir / f"labels_{seq_tensor.shape[-1]}.pth"

    torch.save(seq_tensor, seq_out)
    torch.save(label_tensor, label_out)

    print(f"[OK] Wrote:\n  {seq_out} ({seq_tensor.shape})\n  {label_out} ({label_tensor.shape})")

if __name__ == "__main__":
    main()
