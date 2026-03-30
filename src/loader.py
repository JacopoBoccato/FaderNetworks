import os
import torch
from torch.autograd import Variable
from logging import getLogger
from typing import List

logger = getLogger()
##########################DUMMY TO REMOVE##########################
AVAILABLE_ATTR = ["BinaryAttr"] 


# ============================================================
# Alphabet definitions
# ============================================================

# Normal alphabet: 20 amino acids + '-' padding
ALPHABET_NORMAL = list("ACDEFGHIKLMNPQRSTVWY-")

# Lattice alphabet (used for coarse-grained representations)
ALPHABET_LATTICE = list("CMFILVWYAGTSNQDEHRKP")

# RNA alphabet: A, C, G, U + optional '-' padding
ALPHABET_RNA = list("ACGU-")

def get_alphabet(alphabet_type: str):
    """Return symbol list for the chosen alphabet."""
    if alphabet_type == "normal":
        return ALPHABET_NORMAL
    elif alphabet_type == "lattice":
        return ALPHABET_LATTICE
    elif alphabet_type == "rna":
        return ALPHABET_RNA
    else:
        raise ValueError(f"Unknown alphabet_type: {alphabet_type}. "
                         "Valid types: 'normal', 'lattice', 'rna'.")


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
    Convert a list of sequences (protein, lattice, or RNA) into one-hot encoded tensors.

    Args:
        sequences: list of strings
        alphabet_type: 'normal', 'lattice', or 'rna'
        seq_len: fixed length to pad/truncate all sequences
        pad_token: padding symbol ('-')

    Returns:
        encoded: Tensor [N, n_symbols, seq_len]
        alphabet: list of symbols used
    """
    alphabet = get_alphabet(alphabet_type)
    n_sym = len(alphabet)
    symbol_to_idx = {sym: i for i, sym in enumerate(alphabet)}

    # Normalize RNA sequences (replace T→U)
    if alphabet_type == "rna":
        sequences = [s.upper().replace("T", "U") for s in sequences]

    if seq_len is None:
        seq_len = max(len(s) for s in sequences)

    encoded = torch.zeros((len(sequences), n_sym, seq_len), dtype=torch.float32)

    for n, seq in enumerate(sequences):
        seq = seq.upper()
        for pos, char in enumerate(seq[:seq_len]):
            # unknown characters get encoded as pad token
            idx = symbol_to_idx.get(char, symbol_to_idx.get(pad_token, n_sym - 1))
            encoded[n, idx, pos] = 1.0
        if len(seq) < seq_len:
            pad_idx = symbol_to_idx.get(pad_token, n_sym - 1)
            encoded[n, pad_idx, len(seq):] = 1.0

    return encoded, alphabet


def save_onehot_dataset(sequences, path, alphabet_type='normal', seq_len=None):
    """Encode sequences as one-hot and save to file path."""
    encoded, _ = onehot_encode_sequences(sequences, alphabet_type=alphabet_type, seq_len=seq_len)
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    torch.save(encoded, path)
    return encoded

# ============================================================
# Sequence loading
# ============================================================

def load_sequences(params, alphabet_type="normal", x_type="onehot"):
    alphabet = get_alphabet(alphabet_type)
    expected_amino = len(alphabet)
    logger.info(f"Using '{alphabet_type}' alphabet ({expected_amino} symbols). x_type={x_type}")

    # Load sequence tensor
    seq_path = os.path.join(params.data_path, f"sequences_{params.seq_len}.pth")
    sequences = torch.load(seq_path)

    if x_type == 'continuous':
        # sequences expected as [N, T] float values
        sequences = sequences.float()
        if sequences.dim() == 3 and sequences.size(1) == 1:
            sequences = sequences.squeeze(1)
        if sequences.dim() != 2:
            raise ValueError('Continuous-mode sequences must be 2D (N, seq_len)')
    else:
        # existing one-hot mode
        sequences = sequences.float()

    # Check that all sequences are exactly of length seq_len
    if sequences.shape[-1] != params.seq_len:
        raise ValueError(f"All sequences must be exactly of length {params.seq_len}, "
                         f"but found sequences with length {sequences.shape[-1]}")

    # Load or infer attributes
    attr_path = os.path.join(params.data_path, "attributes.pth")
    label_path = os.path.join(params.data_path, f"labels_{params.seq_len}.pth")

    if os.path.isfile(label_path):
        labels = torch.load(label_path)
        labels = labels.float()
        params.n_attr = labels.size(1)
        logger.info(f"Detected {params.n_attr} attributes from label tensor.")
    else:
        # if no labels, generate random dummy attributes
        N = sequences.size(0)
        if params.label_type == 'continuous':
            params.n_attr = 1
            labels = torch.rand(N, params.n_attr, dtype=torch.float32)
        else:
            params.n_attr = 2
            labels = torch.randint(0, 2, (N, params.n_attr), dtype=torch.float32)
        logger.warning("No labels found; generated random labels for testing.")

    # Save attribute metadata if missing
    if not os.path.isfile(attr_path):
        if params.label_type == 'continuous':
            attributes = [("ContinuousAttr", 1)]
        else:
            attributes = [(f"Attribute{i+1}", int(labels[:, i].unique().numel())) for i in range(params.n_attr)]
        torch.save(attributes, attr_path)
        logger.info(f"Saved inferred attributes to {attr_path}")

    # Split train/valid/test
    N = sequences.size(0)
    n_train = int(0.8 * N)
    n_valid = int(0.1 * N)
    n_test = N - n_train - n_valid

    seq_train, seq_valid, seq_test = torch.split(sequences, [n_train, n_valid, n_test])
    lab_train, lab_valid, lab_test = torch.split(labels, [n_train, n_valid, n_test])

    logger.info(f"Loaded {N} sequences ({n_train} train / {n_valid} valid / {n_test} test)")
    return [seq_train, seq_valid, seq_test], [lab_train, lab_valid, lab_test]


# ============================================================
# Data Sampler
# ============================================================

class DataSampler(object):
    """Samples random batches of one-hot sequences and binary labels."""

    def __init__(self, sequences, labels, params):
        # sequences can be one-hot ([N, C, T]) or continuous ([N, T])
        if getattr(params, 'x_type', 'onehot') == 'continuous':
            if sequences.dim() == 2:
                # already [N, T] continuous
                pass
            elif sequences.dim() == 3 and sequences.size(1) == 1:
                sequences = sequences.squeeze(1)
            else:
                raise ValueError("Unsupported continuous sequence shape for x_type='continuous'")
        else:
            # one-hot mode, ensure [N, C, T]
            pass

        self.sequences = sequences
        self.labels = labels
        self.batch_size = params.batch_size
        self.use_cuda = getattr(params, "cuda", False)
        self._index = 0
        self._order = torch.randperm(self.sequences.size(0))
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

    def eval_batch(self, i, j):
        """
        Return a fixed batch of data between indices [i, j)
        (used for evaluation to ensure determinism).
        """
        batch_seq = self.sequences[i:j]
        batch_labels = self.labels[i:j]
        if self.use_cuda:
            batch_seq = batch_seq.cuda()
            batch_labels = batch_labels.cuda()
        return Variable(batch_seq), Variable(batch_labels)


    def __len__(self):
        return self.sequences.size(0)
