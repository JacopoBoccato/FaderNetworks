#!/usr/bin/env python3
import torch
from pathlib import Path
import pickle
from types import SimpleNamespace
import sys
import numpy as np
import random

# ----------------------------------------------------------------------
# Deterministic setup
# ----------------------------------------------------------------------
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Allow importing from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model import AutoEncoder
from src.loader import onehot_encode_sequences

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def load_params(run_dir: str):
    """Load training parameters from params.pkl"""
    with open(Path(run_dir) / "params.pkl", "rb") as f:
        raw = pickle.load(f)
        if hasattr(raw, "__dict__"):
            return SimpleNamespace(**raw.__dict__)
        elif isinstance(raw, dict):
            return SimpleNamespace(**raw)
        else:
            raise ValueError("Invalid params.pkl format")

def onehot_to_seq(onehot, alphabet):
    """Convert one-hot tensor to amino acid string."""
    if onehot.dim() == 3:
        onehot = onehot.squeeze(0)
    idxs = onehot.argmax(dim=0)
    return "".join(alphabet[int(i)] for i in idxs.cpu().tolist())

def cross_entropy_per_sequence(pred, target):
    """Per-sequence mean cross-entropy."""
    log_probs = torch.log_softmax(pred, dim=1)
    per_token = -(target * log_probs).sum(dim=1)  # (B, L)
    return per_token.mean(dim=1)

# ----------------------------------------------------------------------
# Scoring function (as defined earlier)
# ----------------------------------------------------------------------

def score_sequence(seq):
    """
    Returns the charge-pattern score for a single sequence.
    Positive at positions [0,24], negative at [1,15,17,19].
    """
    indices_pos = [0, 24]
    indices_neg = [1, 15, 17, 19]
    positive_letters = {'H', 'K', 'R'}
    negative_letters = {'D', 'E'}

    score = 0
    seq = seq.upper()

    for idx in indices_pos:
        if 0 <= idx < len(seq):
            if seq[idx] in positive_letters:
                score += 1
            elif seq[idx] in negative_letters:
                score -= 1

    for idx in indices_neg:
        if 0 <= idx < len(seq):
            if seq[idx] in positive_letters:
                score -= 1
            elif seq[idx] in negative_letters:
                score += 1

    return score

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    # === Config ===
    run_dir = "models/lattice_EM/4qrd008urh"   # <-- change to your model folder
    data_dir = Path("data/lattice_EM/processed")          # <-- where sequences_*.pth and labels_*.pth are
    seq_path = data_dir / "sequences_27.pth"
    label_path = data_dir / "labels_27.pth"
    out_path = Path(run_dir) / "sampled_with_scores.txt"

    n_samples = 1000  # number of sequences to sample for inspection

    # === Load params and model ===
    params = load_params(run_dir)
    device = torch.device("cuda" if getattr(params, "cuda", False) and torch.cuda.is_available() else "cpu")

    model = AutoEncoder(params).to(device)
    model_path = Path(run_dir) / "best_rec_ae.pth"
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] Loaded trained model from {model_path}")

    # === Load dataset tensors ===
    x = torch.load(seq_path)
    y = torch.load(label_path)
    print(f"[INFO] Loaded {x.shape[0]} sequences from {seq_path.name}")

    # Truncate or sample
    n_samples = min(n_samples, x.shape[0])
    idxs = torch.arange(n_samples)
    x = x[idxs].to(device)
    y = y[idxs].to(device)
    B, C, L = x.shape
    print(f"[INFO] Sampling {B} sequences of shape ({C}, {L})")

    # === Decode ===
    with torch.no_grad():
        enc = model.encode(x)
        rec_logits = model.decode(enc, y)[-1]
        rec_flip = model.decode(enc, 1.0 - y)[-1]

    # === Compute reconstruction losses ===
    per_seq_loss = cross_entropy_per_sequence(rec_logits, x)

    # === Get alphabet (from params) ===
    _, alphabet = onehot_encode_sequences(["A"], alphabet_type=params.alphabet_type, seq_len=1)

    # === Write output ===
    with open(out_path, "w") as f:
        f.write("# idx\tscore_rec\tscore_flip\trec_loss\torig_label\tflipped_label\tdecoded_seq\tflipped_seq\n")

        for i in range(B):
            rec_seq = onehot_to_seq(rec_logits[i], alphabet)
            flip_seq = onehot_to_seq(rec_flip[i], alphabet)
            score_rec = score_sequence(rec_seq)
            score_flip = score_sequence(flip_seq)
            loss = per_seq_loss[i].item()
            orig_label = y[i].cpu().numpy().tolist()
            flip_label = (1.0 - y[i]).cpu().numpy().tolist()

            f.write(
                f"{i}\t{score_rec}\t{score_flip}\t{loss:.4f}\t"
                f"{orig_label}\t{flip_label}\t{rec_seq}\t{flip_seq}\n"
            )

    print(f"[DONE] Saved decoded sequences with scores to {out_path}")


if __name__ == "__main__":
    main()

