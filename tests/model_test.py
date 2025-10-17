# test_model.py
import torch
from src.model import AutoEncoder

class Params:
    seq_len = 128           # Length of protein sequence
    n_amino = 21            # Alphabet size
    instance_norm = False
    init_fm = 32
    max_fm = 128
    n_layers = 5
    n_skip = 1
    deconv_method = "convtranspose"
    dec_dropout = 0.0
    attr = [("BinaryAttr", 2)]
    n_attr = 2

params = Params()

# Instantiate model
ae = AutoEncoder(params)
print("✅ AutoEncoder initialized")

# Create dummy batch of protein sequences
BATCH = 4
x = torch.randn(BATCH, params.n_amino, params.seq_len)  # [B, 21, 128]
y = torch.randint(0, 2, (BATCH, params.n_attr)).float() # [B, 2]

# Forward pass
enc, dec = ae(x, y)
reconstructed = dec[-1]

print("Input  shape:", x.shape)
print("Output shape:", reconstructed.shape)

# Verify shape consistency
assert reconstructed.shape == x.shape, "Output does not match input shape!"

# Verify gradient flow
loss = torch.nn.functional.mse_loss(reconstructed, x)
loss.backward()
print("✅ Backprop successful, gradient check passed.")
