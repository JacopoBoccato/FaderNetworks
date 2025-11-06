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

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.model import AutoEncoder


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


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    run_dir = "models/lattice_EM/4qrd008urh"
    data_dir = Path("data/lattice_EM/processed")
    seq_path = data_dir / "sequences_27.pth"
    label_path = data_dir / "labels_27.pth"
    out_path = Path(run_dir) / "encoder_outputs.txt"

    print(f"[INFO] Loading model and data from {run_dir}")
    params = load_params(run_dir)
    device = torch.device("cuda" if getattr(params, "cuda", False) and torch.cuda.is_available() else "cpu")

    # === Load model ===
    model = AutoEncoder(params).to(device)
    model_path = Path(run_dir) / "best_rec_ae.pth"
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] Loaded model from {model_path}")

    # === Load dataset ===
    x = torch.load(seq_path)
    y = torch.load(label_path)
    n_samples = min(1000, x.shape[0])
    x = x[:n_samples].to(device)
    y = y[:n_samples].to(device)
    print(f"[INFO] Loaded {n_samples} sequences of shape {x.shape}")

    # === Encode ===
    with torch.no_grad():
        enc_out = model.encode(x)
        if isinstance(enc_out, (tuple, list)):
            latent = enc_out[0]
        elif isinstance(enc_out, dict):
            latent = enc_out.get("latent", list(enc_out.values())[0])
        else:
            latent = enc_out

    if latent.shape[0] != x.shape[0]:
        raise RuntimeError(f"Encoder output batch size mismatch: {latent.shape[0]} vs {x.shape[0]}")

    # === Save ===
    with open(out_path, "w") as f:
        f.write("# idx\tlabel\tlatent_vector\n")
        for i in range(latent.shape[0]):
            label = y[i].cpu().numpy().tolist()
            latent_vec = latent[i].flatten().cpu().numpy()
            latent_str = " ".join(f"{v:.6f}" for v in latent_vec)
            f.write(f"{i}\t{label}\t{latent_str}\n")

    print(f"[DONE] Saved encoder outputs to {out_path}")


if __name__ == "__main__":
    main()

