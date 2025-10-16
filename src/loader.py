import os
import torch
from torch.autograd import Variable
from logging import getLogger
from typing import List

logger = getLogger()

# ============================================================
# Alphabet definitions
# ============================================================

# Normal alphabet: 20 amino acids + '-' padding
ALPHABET_NORMAL = list("ACDEFGHIKLMNPQRSTVWY-")

# Lattice alphabet: ordered by hydrophobicity + '-' padding
ALPHABET_LATTICE = list("CMFILVWYAGTSNQDEHRKP-")

def get_alphabet(alphabet_type: str):
    """Return symbol list for the chosen alphabet."""
    if alphabet_type == "normal":
        return ALPHABET_NORMAL
    elif alphabet_type == "lattice":
        return ALPHABET_LATTICE
    else:
        raise ValueError(f"Unknown alphabet_type: {alphabet_type}")


# ============================================================
# One-hot encoding utilities
# ============================================================

def onehot_encode_sequences(
    sequences: List[str],
    alphabet_type: str = "normal",
    seq_len: int = None,
    pad_token: str = "-",
):
    """
    Convert a list of sequences into one-hot encoded tensors.

    Args:
        sequences: list of strings (protein or lattice sequences)
        alphabet_type: 'normal' or 'lattice'
        seq_len: fixed length to pad/truncate all sequences
        pad_token: padding symbol ('-')

    Returns:
        encoded: Tensor [N, n_amino, seq_len]
        alphabet: list of symbols used
    """
    alphabet = get_alphabet(alphabet_type)
    n_amino = len(alphabet)
    symbol_to_idx = {sym: i for i, sym in enumerate(alphabet)}

    if seq_len is None:
        seq_len = max(len(s) for s in sequences)

    encoded = torch.zeros((len(sequences), n_amino, seq_len), dtype=torch.float32)

    for n, seq in enumerate(sequences):
        for pos, char in enumerate(seq[:seq_len]):
            idx = symbol_to_idx.get(char, symbol_to_idx.get(pad_token, n_amino - 1))
            encoded[n, idx, pos] = 1.0
        if len(seq) < seq_len:
            pad_idx = symbol_to_idx.get(pad_token, n_amino - 1)
            encoded[n, pad_idx, len(seq):] = 1.0

    return encoded, alphabet


def save_onehot_dataset(
    sequences: List[str],
    output_path: str,
    alphabet_type: str = "normal",
    seq_len: int = None,
):
    """
    One-hot encode sequences and save them as a .pth tensor.

    Example:
        save_onehot_dataset(seqs, "data/sequences_256.pth", alphabet_type="normal")
    """
    onehot_tensor, alphabet = onehot_encode_sequences(sequences, alphabet_type, seq_len)
    torch.save(onehot_tensor, output_path)
    print(f"✅ Saved one-hot dataset to {output_path} with shape {tuple(onehot_tensor.shape)}")


# ============================================================
# Sequence loading
# ============================================================

def load_sequences(params, alphabet_type="normal"):
    """
    Load one-hot encoded sequences and binary attribute labels.

    Expected files:
        - sequences_<seq_len>.pth
        - attributes.pth
    """
    alphabet = get_alphabet(alphabet_type)
    expected_amino = len(alphabet)

    logger.info(f"Using '{alphabet_type}' alphabet ({expected_amino} symbols).")
    logger.info(f"Expected input channels: {params.n_amino}")

    # Load sequence tensor
    seq_filename = f"sequences_{params.seq_len}.pth"
    seq_path = os.path.join(params.data_path, seq_filename)
    if not os.path.isfile(seq_path):
        raise FileNotFoundError(f"Sequence file not found: {seq_path}")

    sequences = torch.load(seq_path)  # shape [N, n_amino, seq_len]
    if sequences.dim() != 3 or sequences.size(1) != params.n_amino:
        raise ValueError(
            f"Expected sequences of shape [N, {params.n_amino}, {params.seq_len}], "
            f"but got {tuple(sequences.shape)}"
        )

    sequences = sequences.float()

    # Load attribute labels
    attr_path = os.path.join(params.data_path, "attributes.pth")
    if not os.path.isfile(attr_path):
        raise FileNotFoundError(f"Attribute labels file not found: {attr_path}")
    labels = torch.load(attr_path)
    if labels.size(0) != sequences.size(0):
        raise ValueError("Number of labels does not match number of sequences")

    # Split train / valid / test
    N = sequences.size(0)
    n_train = int(0.8 * N)
    n_valid = int(0.1 * N)
    n_test = N - n_train - n_valid

    seq_train = sequences[:n_train]
    seq_valid = sequences[n_train:n_train + n_valid]
    seq_test = sequences[n_train + n_valid:]
    labels_train = labels[:n_train]
    labels_valid = labels[n_train:n_train + n_valid]
    labels_test = labels[n_train + n_valid:]

    logger.info(
        f"Loaded {N} sequences ({n_train} train / {n_valid} valid / {n_test} test)"
    )

    return [seq_train, seq_valid, seq_test], [labels_train, labels_valid, labels_test]


# ============================================================
# Data Sampler
# ============================================================

class DataSampler(object):
    """Samples random batches of one-hot sequences and binary labels."""

    def __init__(self, sequences, labels, params):
        self.sequences = sequences
        self.labels = labels
        self.batch_size = params.batch_size
        self.use_cuda = getattr(params, "cuda", False)
        self._index = 0
        self._order = torch.randperm(self.sequences.size(0))

    def _shuffle_data(self):
        self._order = torch.randperm(self.sequences.size(0))
        self._index = 0

    def next_batch(self):
        if self._index + self.batch_size > self.sequences.size(0):
            self._shuffle_data()
        idx = self._order[self._index:self._index + self.batch_size]
        self._index += self.batch_size
        batch_seq = self.sequences[idx]
        batch_labels = self.labels[idx]
        if self.use_cuda:
            batch_seq = batch_seq.cuda()
            batch_labels = batch_labels.cuda()
        return Variable(batch_seq), Variable(batch_labels)

    def __len__(self):
        return self.sequences.size(0)
