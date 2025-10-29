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

# allow importing from project root
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

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


def load_seqs(file_path):
    """Read score/sequence pairs and return only sequences."""
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return [lines[i] for i in range(1, len(lines), 2)]


def onehot_to_seq(onehot, alphabet):
    if onehot.dim() == 3:
        onehot = onehot.squeeze(0)
    idxs = onehot.argmax(dim=0)
    return "".join(alphabet[int(i)] for i in idxs.cpu().tolist())


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # === Config ===
    run_dir = "models/lattice_dense_minimal/ijfu5kcvop"  # <-- adjust this
    model_path = Path(run_dir) / "best_rec_ae.pth"
    data_dir = Path("data/lattice")
    file_name = "LatticeA.txt"  # <-- choose "LatticeA.txt" or "LatticeB.txt"
    output_path = project_root / f"decoded_flipped_{file_name}"

    # === Load model ===
    params = load_params(run_dir)
    device = torch.device("cuda" if getattr(params, "cuda", False) and torch.cuda.is_available() else "cpu")

    model = AutoEncoder(params).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # === Load sequences ===
    seqs = load_seqs(data_dir / file_name)
    print(f"[INFO] Loaded {len(seqs)} sequences from {file_name}")

    # === Prepare attributes ===
    n_attr = getattr(params, "n_attr", 2)
    attr = torch.zeros((len(seqs), n_attr), dtype=torch.float32)
    attr[:, 0] = 1.0  # mark all as belonging to the first attribute

    # === Encode sequences ===
    x, alphabet = onehot_encode_sequences(
        seqs,
        alphabet_type=getattr(params, "alphabet_type", "lattice"),
        seq_len=params.seq_len,
    )
    x, attr = x.to(device), attr.to(device)

    with torch.no_grad():
        enc = model.encode(x)
        rec_logits = model.decode(enc, attr)[-1]
        rec_flip = model.decode(enc, 1.0 - attr)[-1]

    # === Write to file ===
    with open(output_path, "w") as f:
        f.write("# Original\tDecoded\tFlipped\n")
        for i, seq in enumerate(seqs):
            rec_seq = onehot_to_seq(rec_logits[i], alphabet)
            flip_seq = onehot_to_seq(rec_flip[i], alphabet)
            f.write(f"{seq}\t{rec_seq}\t{flip_seq}\n")

    print(f"[DONE] Saved decoded and flipped sequences to: {output_path}")


if __name__ == "__main__":
    main()
