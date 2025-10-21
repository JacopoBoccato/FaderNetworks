"""
Sample protein sequences from a trained FaderNetworks autoencoder.

Compatible with the 1D sequence version (no image code).
Works with PyTorch 2.6+ and supports both full-model and state_dict checkpoints.
"""

import os
import torch
from torch.nn import ModuleList
from src.model import AutoEncoder
from src.loader import load_sequences

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = "data"
MODEL_PATH = "models/protein_seq_exp/6798pfqcko/best_rec_ae.pth"  # adjust if needed
SEQ_LEN = 128
N_AMINO = 21
ALPHABET_TYPE = "normal"

# ---------------------------------------------------------------------------
# Safe PyTorch 2.6+ model loading setup
# ---------------------------------------------------------------------------
torch.serialization.add_safe_globals([AutoEncoder, ModuleList])

print("Loading trained model...")

# Create minimal params object for AE construction if we need it
params = type("Params", (), {
    "seq_len": SEQ_LEN,
    "n_amino": N_AMINO,
    "init_fm": 32,
    "max_fm": 512,
    "n_layers": 6,
    "n_skip": 0,
    "deconv_method": "convtranspose",
    "instance_norm": False,
    "dec_dropout": 0.0,
    "attr": [("", 2)],
    "n_attr": 2,
})()

# Try to load checkpoint robustly (handles both torch.save(ae) and state_dict)
try:
    obj = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        ae = AutoEncoder(params)
        ae.load_state_dict(obj)
        print("Loaded state_dict checkpoint.")
    else:
        ae = obj
        print("Loaded full AutoEncoder object.")
except Exception as e:
    raise RuntimeError(f"Failed to load checkpoint {MODEL_PATH}: {e}")

ae.eval()
print("✅ Model ready for inference.")

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
print("Loading dataset...")
class DummyParams:
    seq_len = SEQ_LEN
    n_amino = N_AMINO
    data_path = DATA_PATH

dummy_params = DummyParams()
data, labels = load_sequences(dummy_params, alphabet_type=ALPHABET_TYPE)
test_x, test_y = data[2][:5], labels[2][:5]  # use 5 test sequences

print(f"Dataset loaded. Shape: {tuple(test_x.shape)}")

# ---------------------------------------------------------------------------
# Reconstruction sampling
# ---------------------------------------------------------------------------
print("\nReconstructing sequences...")
with torch.no_grad():
    _, dec_outputs = ae(test_x, test_y)
    recon = dec_outputs[-1]

# ---------------------------------------------------------------------------
# Convert one-hot tensors back to letters (diagnostic-safe)
# ---------------------------------------------------------------------------
alphabet = "ACDEFGHIKLMNPQRSTVWY-"

print("\nInspecting reconstruction tensor shapes:")
print(f"Original batch: {test_x.shape}")
print(f"Reconstruction batch: {recon.shape}")
print(f"Reconstruction stats: min={recon.min().item():.4f}, max={recon.max().item():.4f}, mean={recon.mean().item():.4f}")

# Defensive reshape (ensure batch dimension)
if recon.dim() == 2:
    recon = recon.unsqueeze(0)

orig_indices = test_x.argmax(1)
recon_indices = recon.argmax(1)

print("\nSample reconstructions:")
for i in range(min(3, len(orig_indices))):
    try:
        orig_seq = "".join(alphabet[j] for j in orig_indices[i].tolist())
        recon_seq = "".join(alphabet[j] for j in recon_indices[i].tolist())
        print(f"[{i}] Original:     {orig_seq}")
        print(f"    Reconstructed: {recon_seq}")
    except Exception as e:
        print(f"[{i}] ⚠️ Could not print sequence: {e}")

# ---------------------------------------------------------------------------
# Attribute flipping (optional Fader-style manipulation)
# ---------------------------------------------------------------------------
print("\nFlipping attributes...")
flipped_y = 1 - test_y  # binary inversion for demo
with torch.no_grad():
    _, flipped_dec = ae(test_x, flipped_y)
    flipped_recon = flipped_dec[-1]
flipped_indices = flipped_recon.argmax(1)

for i in range(min(3, len(flipped_indices))):
    flip_seq = "".join(alphabet[j] for j in flipped_indices[i].tolist())
    print(f"    Flipped:       {flip_seq}")

# ---------------------------------------------------------------------------
# Quantitative reconstruction accuracy (optional sanity check)
# ---------------------------------------------------------------------------
recon_acc = (orig_indices == recon_indices).float().mean().item()
print(f"\n🧠 Reconstruction accuracy: {recon_acc * 100:.2f}%")
print("✅ Sampling complete.")
