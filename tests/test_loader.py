# test_loader.py
from src.loader import onehot_encode_sequences, save_onehot_dataset, get_alphabet
import torch

print("=== TEST 1: Alphabet definitions ===")
print("Normal:", "".join(get_alphabet("normal")))
print("Lattice:", "".join(get_alphabet("lattice")))
print()

print("=== TEST 2: One-hot encoding ===")
seqs = ["ACDEFGHIKLMNPQRSTVWY-", "CMFILVWYAGTSNQDEHRKP-"]
encoded, alphabet = onehot_encode_sequences(seqs, alphabet_type="normal", seq_len=22)

print("Alphabet size:", len(alphabet))
print("Encoded tensor shape:", encoded.shape)
print("Sum across amino acid channels (should all be 1):")
print(encoded[0].sum(dim=0))
print()

print("=== TEST 3: Save and reload ===")
save_onehot_dataset(seqs, "data/sequences_22.pth", alphabet_type="normal", seq_len=22)

# dummy labels
labels = torch.randint(0, 2, (len(seqs), 2)).float()
torch.save(labels, "data/attributes.pth")

# minimal params class for testing load_sequences
class Params:
    seq_len = 22
    n_amino = 21
    batch_size = 2
    data_path = "data"
    cuda = False

from src.loader import load_sequences, DataSampler

params = Params()
data, attr = load_sequences(params, alphabet_type="normal")
print("Train data shape:", data[0].shape)
print("Train label shape:", attr[0].shape)

sampler = DataSampler(data[0], attr[0], params)
x, y = sampler.next_batch()
print("Batch shapes:", x.shape, y.shape)
print("\n✅ All loader tests completed successfully!")
