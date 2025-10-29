import torch
import os

data_path = "data/lattice/processed"
seq_path = os.path.join(data_path, "sequences_27.pth")
label_path = os.path.join(data_path, "labels_27.pth")

# Load sequences to get N
sequences = torch.load(seq_path)
N = sequences.size(0)

# Build binary one-hot attributes
half = N // 2
attrA = torch.zeros((half, 2))
attrA[:, 0] = 1
attrB = torch.zeros((N - half, 2))
attrB[:, 1] = 1
labels = torch.cat([attrA, attrB], dim=0)

# Save new labels
torch.save(labels, label_path)
print(f"[OK] Wrote {label_path} with shape {labels.shape}")

