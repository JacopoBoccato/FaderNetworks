import os
import random
import numpy as np
import torch

from src.loader import onehot_encode_sequences, get_alphabet

# ============================================================
# Configuration (must match train.py arguments)
# ============================================================
seq_len = 128
n_train = 1000
n_valid = 100
alphabet_type = "normal"   # can also be "lattice"
pad_token = "-"

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# ============================================================
# Random sequence generator
# ============================================================
def make_random_sequences(n_samples, seq_len, alphabet_type):
    """Generate random sequences and introduce one random pad ('-') per sequence."""
    alphabet = get_alphabet(alphabet_type)
    alphabet_no_pad = [a for a in alphabet if a != pad_token]
    sequences = []

    for _ in range(n_samples):
        # Random sequence excluding pad token
        seq = "".join(random.choices(alphabet_no_pad, k=seq_len))
        # Introduce one random pad token ('-') → acts as a random zero
        rand_pos = random.randint(0, seq_len - 1)
        seq = seq[:rand_pos] + pad_token + seq[rand_pos + 1:]
        sequences.append(seq)

    return sequences

# ============================================================
# Dataset builder
# ============================================================
def make_dataset(n_samples, seq_len, alphabet_type):
    sequences = make_random_sequences(n_samples, seq_len, alphabet_type)
    encoded, alphabet = onehot_encode_sequences(sequences, alphabet_type, seq_len)
    labels = torch.randint(0, 2, (n_samples, 2), dtype=torch.float32)
    return encoded, labels

# ============================================================
# Generate and save both .npy and .pth formats
# ============================================================
print("🧬 Generating dummy training data...")
train_X, train_y = make_dataset(n_train, seq_len, alphabet_type)

print("🧪 Generating dummy validation data...")
valid_X, valid_y = make_dataset(n_valid, seq_len, alphabet_type)

# Save as NumPy arrays (optional)
np.save(os.path.join(data_dir, "train_sequences.npy"), train_X.numpy())
np.save(os.path.join(data_dir, "train_labels.npy"), train_y.numpy())
np.save(os.path.join(data_dir, "valid_sequences.npy"), valid_X.numpy())
np.save(os.path.join(data_dir, "valid_labels.npy"), valid_y.numpy())

# Save as PyTorch tensors (.pth format)
torch.save(train_X, os.path.join(data_dir, f"sequences_{seq_len}.pth"))
torch.save(train_y, os.path.join(data_dir, f"labels_{seq_len}.pth"))
torch.save(valid_X, os.path.join(data_dir, f"valid_sequences_{seq_len}.pth"))
torch.save(valid_y, os.path.join(data_dir, f"valid_labels_{seq_len}.pth"))

print("✅ Dummy dataset created and saved in both .npy and .pth formats!")
print(f"   Train shape: {train_X.shape}, Valid shape: {valid_X.shape}")
