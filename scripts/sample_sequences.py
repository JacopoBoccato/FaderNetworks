import torch
from pathlib import Path
import pickle
from types import SimpleNamespace
import sys
import torch.nn.functional as F
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

# allow importing from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model import AutoEncoder
from src.loader import onehot_encode_sequences


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def load_params(run_dir: str):
    with open(Path(run_dir) / "params.pkl", "rb") as f:
        raw = pickle.load(f)
        if hasattr(raw, "__dict__"):
            return SimpleNamespace(**raw.__dict__)
        elif isinstance(raw, dict):
            return SimpleNamespace(**raw)
        else:
            raise ValueError("Invalid params.pkl format")


def load_seqs(file_path, max_count=None):
    """Read score/sequence pairs and return only sequences."""
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    seqs = [lines[i] for i in range(1, len(lines), 2)]
    # sort to ensure deterministic ordering
    return seqs if max_count is None else seqs[:max_count]


def onehot_to_seq(onehot, alphabet):
    if onehot.dim() == 3:
        onehot = onehot.squeeze(0)
    idxs = onehot.argmax(dim=0)
    return "".join(alphabet[int(i)] for i in idxs.cpu().tolist())


def cross_entropy_per_sequence(pred, target):
    """
    pred: (B, C, L)
    target: (B, C, L)
    Returns vector of per-sequence mean cross-entropy values.
    """
    log_probs = torch.log_softmax(pred, dim=1)
    per_token = -(target * log_probs).sum(dim=1)  # (B, L)
    return per_token.mean(dim=1)  # (B,)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    # === Config ===
    run_dir = "models/lattice_dense_minimal/ijfu5kcvop"   # <-- adjust to your folder
    model_path = Path(run_dir) / "best_rec_ae.pth"
    data_dir = Path("data/lattice")
    out_path = Path(run_dir) / "validation_results.txt"
    n_per_class = 50

    # === Load params & model ===
    params = load_params(run_dir)
    device = torch.device("cuda" if getattr(params, "cuda", False) and torch.cuda.is_available() else "cpu")

    model = AutoEncoder(params).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] Loaded model from {model_path}")

    # === Load and encode all sequences once ===
    allA = load_seqs(data_dir / "LatticeA.txt")
    allB = load_seqs(data_dir / "LatticeB.txt")
    allseqs = allA + allB
    all_onehot, alphabet = onehot_encode_sequences(
        allseqs,
        alphabet_type=getattr(params, "alphabet_type", "lattice"),
        seq_len=params.seq_len,
    )

    # === Compute deterministic baseline ONCE ===
    all_onehot = all_onehot.float().cpu()
    freq = all_onehot.mean(dim=0, keepdim=True)
    offset = torch.arange(freq.shape[1]).view(1, -1, 1) * 1e-12  # deterministic tie-break
    freq = freq + offset
    baseline_pred = torch.zeros_like(freq).scatter_(
        1, freq.argmax(dim=1, keepdim=True), 1.0
    )

    # === Compute dummy reconstruction loss ONCE (global baseline) ===
    with torch.no_grad():
        log_probs = torch.log_softmax(baseline_pred, dim=1)
        per_token = -(all_onehot * log_probs).sum(dim=1)
        dummy_loss = per_token.mean().item()

    print(f"[INFO] Computed deterministic dummy baseline loss over full dataset: {dummy_loss:.4f}")
    print(f"[INFO] Computed baseline amino-acid frequencies from {len(allseqs)} sequences")

    # === Prepare evaluation subset ===
    seqs_a = allA[:n_per_class]
    seqs_b = allB[:n_per_class]
    raw = seqs_a + seqs_b

    n_attr = getattr(params, "n_attr", 2)
    attr = torch.zeros((len(raw), n_attr), dtype=torch.float32)
    attr[:n_per_class, 0] = 1.0
    attr[n_per_class:, 1] = 1.0

    x, _ = onehot_encode_sequences(
        raw,
        alphabet_type=getattr(params, "alphabet_type", "lattice"),
        seq_len=params.seq_len,
    )
    x, attr = x.to(device), attr.to(device)
    B, C, L = x.shape
    print(f"[INFO] Evaluating {B} sequences of shape ({C}, {L})")

    # === Encode / Decode ===
    with torch.no_grad():
        enc = model.encode(x)
        rec_logits = model.decode(enc, attr)[-1]
        rec_flip = model.decode(enc, 1.0 - attr)[-1]

    # === Reconstruction losses (model only) ===
    per_seq_loss = cross_entropy_per_sequence(rec_logits, x)
    mean_rec_loss = per_seq_loss.mean().item()

    print(f"[RESULT] Mean reconstruction loss (model): {mean_rec_loss:.4f}")
    print(f"[RESULT] Dummy baseline loss: {dummy_loss:.4f}")

    # === Write sequences + losses to file ===
    with open(out_path, "w") as f:
        f.write("# idx\tattr_orig\tattr_flip\trec_loss\tbase_loss\torig_seq\treconstructed_seq\tflipped_seq\n")
        for i in range(B):
            orig_seq = raw[i]
            rec_seq = onehot_to_seq(rec_logits[i], alphabet)
            flip_seq = onehot_to_seq(rec_flip[i], alphabet)
            orig_attr = attr[i].cpu().numpy().tolist()
            flip_attr = (1.0 - attr[i]).cpu().numpy().tolist()
            rec_l = per_seq_loss[i].item()
            f.write(f"{i}\t{orig_attr}\t{flip_attr}\t{rec_l:.4f}\t{dummy_loss:.4f}\t{orig_seq}\t{rec_seq}\t{flip_seq}\n")

    print(f"[DONE] Saved {B} sequence triplets with per-sequence losses to {out_path}")
    print("Alphabet:", alphabet)


if __name__ == "__main__":
    main()
