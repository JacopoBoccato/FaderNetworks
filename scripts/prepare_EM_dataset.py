#!/usr/bin/env python3
import os
from pathlib import Path
import torch
import argparse
import sys

# Add root directory to import src/loader
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.loader import onehot_encode_sequences

ALPHABET_TYPE = "lattice"


# ---------------------------------------------------------------------------
# SCORING AND FILTERING FUNCTION
# ---------------------------------------------------------------------------
def score_and_filter(strings):
    """
    Scores sequences based on position-specific charge rules.
    Keeps only sequences with scores -6, -5, 5, or 6.
    Labels:
        5 or 6  -> [0, 1]
       -5 or -6 -> [1, 0]
    """
    # Rule parameters
    indices_pos = [0, 24]
    indices_neg = [1, 15, 17, 19]
    positive_letters = {'H', 'K', 'R'}
    negative_letters = {'D', 'E'}

    filtered_sequences = []
    labels = []

    for s in strings:
        score = 0

        # Favor positive residues at indices_pos
        for idx in indices_pos:
            if 0 <= idx < len(s):
                if s[idx] in positive_letters:
                    score += 1
                elif s[idx] in negative_letters:
                    score -= 1

        # Favor negative residues at indices_neg
        for idx in indices_neg:
            if 0 <= idx < len(s):
                if s[idx] in positive_letters:
                    score -= 1
                elif s[idx] in negative_letters:
                    score += 1

        # Keep only strongly scored sequences
        if score in (-6, -5, 5, 6):
            filtered_sequences.append(s)
            if score > 0:
                labels.append([0, 1])  # positive pattern
            else:
                labels.append([1, 0])  # negative pattern

    return filtered_sequences, torch.tensor(labels, dtype=torch.float32)


# ---------------------------------------------------------------------------
# READ SEQUENCES FUNCTION
# ---------------------------------------------------------------------------
def read_sequences(path: Path):
    """Reads a file where each line is a sequence."""
    with open(path, "r") as f:
        seqs = [l.strip().upper() for l in f if l.strip()]
    if not seqs:
        raise ValueError(f"No sequences found in {path}")
    return seqs


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Filter, label, and encode protein sequences based on charge pattern rules."
    )
    parser.add_argument("--file", type=str, required=True, help="Path to input sequence file")
    parser.add_argument("--outdir", type=str, required=True, help="Directory to save output .pth files")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read input sequences
    seqs = read_sequences(Path(args.file))
    print(f"[INFO] Read {len(seqs)} sequences from {args.file}")

    # Score and filter
    filtered_seqs, label_tensor = score_and_filter(seqs)
    print(f"[INFO] Filtered {len(filtered_seqs)} sequences matching score criteria (-6,-5,5,6)")

    if not filtered_seqs:
        print("[WARN] No sequences passed the filter. Exiting.")
        return

    # One-hot encode filtered sequences
    seq_tensor, alphabet = onehot_encode_sequences(
        filtered_seqs,
        alphabet_type=ALPHABET_TYPE,
        seq_len=None  # auto-compute max length
    )

    print(f"[INFO] One-hot encoded shape: {seq_tensor.shape}")
    print(f"[INFO] Labels shape: {label_tensor.shape}")

    # Save tensors
    seq_out = outdir / f"sequences_{seq_tensor.shape[-1]}.pth"
    label_out = outdir / f"labels_{seq_tensor.shape[-1]}.pth"

    torch.save(seq_tensor, seq_out)
    torch.save(label_tensor, label_out)

    print(f"[OK] Saved:\n  {seq_out} ({seq_tensor.shape})\n  {label_out} ({label_tensor.shape})")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

