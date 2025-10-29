#!/usr/bin/env python3
"""
Compute column-wise norms of the first decoder layer.

For each input (latent or attribute):
- compute L2 norm of its 150 outgoing weights.
"""

import torch
from pathlib import Path
import pickle
from types import SimpleNamespace
import sys
import numpy as np

# Allow src imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.model import AutoEncoder


def load_params(run_dir: str):
    """Load params.pkl as SimpleNamespace."""
    with open(Path(run_dir) / "params.pkl", "rb") as f:
        raw = pickle.load(f)
        if hasattr(raw, "__dict__"):
            return SimpleNamespace(**raw.__dict__)
        elif isinstance(raw, dict):
            return SimpleNamespace(**raw)
        else:
            raise ValueError("Invalid params.pkl format")


def main():
    # === Adjust paths ===
    run_dir = "models/lattice_dense_minimal/nwkoqpm08u"
    model_path = Path(run_dir) / "best_rec_ae.pth"
    out_path = Path(run_dir) / "decoder_column_norms.txt"

    # === Load model ===
    print(f"[INFO] Loading model from {model_path}")
    params = load_params(run_dir)
    device = torch.device("cuda" if getattr(params, "cuda", False) and torch.cuda.is_available() else "cpu")

    model = AutoEncoder(params).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # === Find first Linear layer in decoder ===
    first_linear = None
    for layer in model.decoder:
        if isinstance(layer, torch.nn.Linear):
            first_linear = layer
            break

    if first_linear is None:
        print("[ERROR] No Linear layer found in decoder.")
        return

    W = first_linear.weight.data.cpu().numpy()  # shape (150, 82)
    attr_dim = getattr(params, "n_attr", 2)
    z_dim = W.shape[1] - attr_dim

    print(f"[INFO] First decoder layer shape: {W.shape}")
    print(f"       latent_dim={z_dim}, attr_dim={attr_dim}\n")

    # === Compute column norms ===
    column_norms = np.linalg.norm(W, axis=0)  # shape (82,)

    latent_norms = column_norms[:z_dim]
    attr_norms = column_norms[-attr_dim:]

    mean_latent = latent_norms.mean()
    attr1, attr2 = attr_norms

    print("[RESULTS]")
    print(f"Mean latent column norm: {mean_latent:.6f}")
    print(f"Attr1 column norm      : {attr1:.6f}")
    print(f"Attr2 column norm      : {attr2:.6f}")
    print(f"Attr1/latent ratio     : {attr1 / (mean_latent + 1e-8):.4f}")
    print(f"Attr2/latent ratio     : {attr2 / (mean_latent + 1e-8):.4f}")

    # === Save detailed norms ===
    with open(out_path, "w") as f:
        f.write("# Column-wise L2 norms of decoder first layer\n")
        f.write(f"# Shape {W.shape}\n")
        f.write(f"# latent_dim={z_dim}, attr_dim={attr_dim}\n\n")
        for j, norm in enumerate(column_norms):
            kind = "latent" if j < z_dim else f"attr{j - z_dim + 1}"
            f.write(f"{j:03d} {kind:8s} {norm:.6f}\n")
        f.write(f"\nMean latent norm: {mean_latent:.6f}\n")
        f.write(f"Attr1 norm: {attr1:.6f}\n")
        f.write(f"Attr2 norm: {attr2:.6f}\n")

    print(f"\n[INFO] Column norms written to {out_path}")


if __name__ == "__main__":
    main()

