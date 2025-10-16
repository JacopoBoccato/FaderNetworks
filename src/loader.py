import os
import torch
from torch.autograd import Variable

class DataSampler(object):
    def __init__(self, sequences, labels, params):
        """
        DataSampler now handles protein sequence data instead of images.
        It takes a tensor of sequences (shape [N, n_amino, seq_len]) and 
        a tensor of binary labels (shape [N, n_attrs]), along with params.
        """
        self.sequences = sequences          # store all sequences (was 'images')
        self.labels = labels                # store all labels (was 'attributes')
        self.n_samples = sequences.size(0)  # total number of sequences
        self.batch_size = params.batch_size
        self.use_cuda = hasattr(params, 'cuda') and params.cuda  # True if CUDA available and enabled

        # Initialize index and order for shuffling
        self._index = 0
        self._order = torch.randperm(self.n_samples)  # random initial order

    def _shuffle_data(self):
        """Shuffle the order of the data (called when an epoch finishes)."""
        self._order = torch.randperm(self.n_samples)
        self._index = 0

    def next_batch(self):
        """
        Fetch the next batch of sequences and labels. If at end of data, reshuffle.
        Returns: (Variable(batch_seq), Variable(batch_labels)), moved to GPU if available.
        """
        # Shuffle for a new epoch if we've exhausted the current order
        if self._index + self.batch_size > self.n_samples:
            self._shuffle_data()
        # Take next batch indices
        indices = self._order[self._index : self._index + self.batch_size]
        self._index += self.batch_size

        # Slice the sequences and labels tensors to get the batch
        batch_seq = self.sequences[indices]
        batch_labels = self.labels[indices]
        # Move to GPU if CUDA is available
        if self.use_cuda:
            batch_seq = batch_seq.cuda()
            batch_labels = batch_labels.cuda()
        # Wrap in torch.autograd.Variable (for PyTorch 0.4 and later, this is optional 
        # as Tensors are autograd-enabled, but we keep it for consistency with original code)
        batch_seq = Variable(batch_seq)
        batch_labels = Variable(batch_labels)
        return batch_seq, batch_labels

    # (Optional) support iteration protocol if needed
    def __iter__(self):
        return self
    def __next__(self):
        return self.next_batch()
    def __len__(self):
        return self.n_samples

def load_sequences(params):
    """
    Load one-hot encoded protein sequences and binary attribute labels.
    Replaces image loading with sequence loading.
    """
    seq_len = params.seq_len   # expected sequence length
    n_amino = params.n_amino   # expected number of amino acid categories (one-hot depth)

    # Construct the file path for sequences based on seq_len (e.g., "sequences_100.pth")
    seq_filename = f"sequences_{seq_len}.pth"
    seq_path = os.path.join(params.data_path, seq_filename)
    if not os.path.isfile(seq_path):
        raise FileNotFoundError(f"Sequence file not found: {seq_path}")
    # Load all sequences (shape [N, n_amino, seq_len])
    sequences = torch.load(seq_path)
    # Validate the shape of loaded sequences
    if sequences.dim() != 3 or sequences.size(1) != n_amino or sequences.size(2) != seq_len:
        raise ValueError(f"Loaded sequences tensor has shape {tuple(sequences.shape)}, expected (*, {n_amino}, {seq_len}).")
    # Convert sequence data to float (one-hot encoded 0/1 values as float32 for model input)
    sequences = sequences.float()

    # Load attribute labels (binary) from attributes.pth
    attr_path = os.path.join(params.data_path, "attributes.pth")
    if not os.path.isfile(attr_path):
        raise FileNotFoundError(f"Attribute labels file not found: {attr_path}")
    labels = torch.load(attr_path)
    # Ensure labels have matching number of samples N
    if labels.size(0) != sequences.size(0):
        raise ValueError(f"Number of sequences ({sequences.size(0)}) does not match number of labels ({labels.size(0)}).")
    # (No change to label values: assume already 0/1 binary as in original attributes.pth)

    # **Train/Validation/Test Split** – keep same logic as original:
    N = sequences.size(0)
    # Determine split sizes. For example, using 80/10/10 split by default (or original dataset-specific counts).
    # Here we use proportional splits similar to CelebA (approximately 80% train, 10% val, 10% test):
    n_train = int(0.8 * N)
    n_valid = int(0.1 * N)
    n_test  = N - n_train - n_valid  # remainder to test
    # Slice the data into splits
    sequences_train = sequences[:n_train]
    labels_train = labels[:n_train]
    sequences_valid = sequences[n_train : n_train + n_valid]
    labels_valid = labels[n_train : n_train + n_valid]
    sequences_test = sequences[n_train + n_valid : ]
    labels_test = labels[n_train + n_valid : ]

    # (The slicing above assumes data is ordered similarly to original dataset. 
    # If a specific partition file or indices are provided, use that for splitting instead.)

    # Return sequences and labels splits as lists/tuples for train, val, test
    return [sequences_train, sequences_valid, sequences_test], [labels_train, labels_valid, labels_test]
