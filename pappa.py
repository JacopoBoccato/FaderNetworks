#!/usr/bin/env python3
# ------------------------------------------------------------
# Debug reconstruction loss and tensor alignment using .pth data
# ------------------------------------------------------------
import torch
from pathlib import Path
from types import SimpleNamespace
import pickle

from src.model import AutoEncoder, sequence_cross_entropy

# ------------------------------------------------------------
# 1. User paths
# ------------------------------------------------------------
run_dir = "/home/ipht/jboccato/FADER_TEST/FaderNetworks/models/lattice_dense_minimal/lb05np9jzr"          # directory with params.pkl and model weights
data_path = "/home/ipht/jboccato/FADER_TEST/FaderNetworks/data/lattice/processed/sequences_27.pth"  # <-- your .pth file with input data

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# 2. Load params and model
# ------------------------------------------------------------
with open(Path(run_dir) / "params.pkl", "rb") as f:
    raw = pickle.load(f)
params = SimpleNamespace(**raw.__dict__) if hasattr(raw, "__dict__") else SimpleNamespace(**raw)

ae = AutoEncoder(params).to(device)
ae.eval()

# optionally load trained weights
weights_path = Path(run_dir) / "best_rec_ae.pth"
if weights_path.exists():
    ae.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded AE weights from {weights_path}")
else:
    print("⚠️ No trained weights found — using random initialization.")

# ------------------------------------------------------------
# 3. Load your .pth data file
# ------------------------------------------------------------
loaded = torch.load(data_path, map_location=device)
if isinstance(loaded, (list, tuple)) and len(loaded) >= 1:
    batch_x = loaded[0]
    batch_y = loaded[1] if len(loaded) > 1 else None
else:
    batch_x = loaded
    batch_y = None

batch_x = batch_x.float().to(device)
print("Loaded data:", batch_x.shape)

# Take only a few sequences if it’s large
batch_x = batch_x[:4]

# ------------------------------------------------------------
# 4. Forward pass
# ------------------------------------------------------------
with torch.no_grad():
    enc_outs, dec_outs = ae(batch_x, batch_y)
    recon = dec_outs[-1]

# ------------------------------------------------------------
# 5. Diagnostics
# ------------------------------------------------------------
B, V, T = recon.shape
print("--------------------------------------------------")
print("x shape      :", batch_x.shape)
print("recon shape  :", recon.shape)
print("min/max x    :", batch_x.min().item(), batch_x.max().item())
print("min/max recon:", recon.min().item(), recon.max().item())

corr = torch.corrcoef(torch.stack([
    recon.detach().view(-1),
    batch_x.view(-1).float()
]))
print("x corr with recon logits:\n", corr)

ce_normal = sequence_cross_entropy(recon, batch_x)
ce_flipped = sequence_cross_entropy(recon.permute(0,2,1), batch_x)
ce_identity = sequence_cross_entropy(batch_x, batch_x)

print("CrossEntropy normal :", ce_normal.item())
print("CrossEntropy flipped:", ce_flipped.item())
print("CrossEntropy(x,x)   :", ce_identity.item())

agreement = (recon.argmax(1) == batch_x.argmax(1)).float().mean().item()
print("Top-1 agreement ratio:", agreement)

probs = torch.softmax(recon, dim=1)
entropy = -(probs * probs.log()).sum(dim=1).mean().item()
print("Mean per-position entropy:", entropy)
print("--------------------------------------------------")

