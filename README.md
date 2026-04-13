# FaderNetworks for Biological Sequences

This repository adapts the original Fader Networks codebase to biological-sequence style data and also keeps a small synthetic linear / Branch-B research area in the root directory.

The repository is best understood as three layers:

- `src/`: reusable model, training, loading, evaluation, and observable code
- root scripts: training entrypoints, synthetic experiments, plotting, and legacy upstream utilities
- `scripts/`: dataset preparation and checkpoint-inspection helpers

For a file-by-file guide to the Python code outside `src/`, see [docs/non_src_python_guide.md](docs/non_src_python_guide.md).

## Current Workflows

### 1. Standard sequence training

Use [`train.py`](train.py) with the modules in [`src/`](src/).

This is the canonical path for training dense sequence Fader models on:

- one-hot sequence inputs with binary / categorical labels
- one-hot sequence inputs with continuous labels
- continuous inputs with continuous labels

The core implementation lives in:

- [`src/model.py`](src/model.py): `AutoEncoder`, `LatentDiscriminator`, optional extra discriminators
- [`src/training.py`](src/training.py): alternating discriminator / autoencoder updates
- [`src/loader.py`](src/loader.py): dataset loading and `DataSampler`
- [`src/utils.py`](src/utils.py): experiment setup, optimizer parsing, lambda scheduling

### 2. Dataset preparation

The main preprocessing helpers live in [`scripts/`](scripts):

- [`scripts/prepare_EM_dataset.py`](scripts/prepare_EM_dataset.py): build a lattice / protein dataset with binary labels derived from charge-pattern rules
- [`scripts/process_rna_sequences.py`](scripts/process_rna_sequences.py): parse RNA records with taxonomy and save tensors plus metadata

Important: `train.py` loads datasets named `sequences_<seq_len>.pth` and `labels_<seq_len>.pth` from `--data_path`. The RNA preprocessing script currently writes `rna_sequences_<L>.pth` and `rna_labels_<L>.pth`, so those outputs need to be renamed or adapted before they are consumed by `train.py`.

### 3. Synthetic linear / Branch-B experiments

There are two root-level theory scripts:

- [`phase_diagram.py`](phase_diagram.py): main strict-linear theory-vs-training comparison script
- [`create_dummy_dataset.py`](create_dummy_dataset.py): near-duplicate synthetic experiment script with a misleading name

These are useful for research experiments, but they are not the main sequence-training entrypoint.

## Repository Layout

```text
.
├── src/          reusable implementation
├── scripts/      dataset prep and checkpoint utilities
├── tests/        smoke-style tests
├── jobs/         PBS launchers
├── models/       experiment outputs
├── results/      saved result artifacts
├── train.py      standard training entrypoint
├── phase_diagram.py
├── create_dummy_dataset.py
├── plot_branch_b_results.py
├── plot_observables.py
├── decoder_weights.py
├── classifier.py
└── interpolate.py
```

## Data Format Expected by `train.py`

Place tensors under `--data_path` using these names:

- `sequences_<seq_len>.pth`
- `labels_<seq_len>.pth`
- optional `attributes.pth`

Supported tensor conventions:

- `x_type=onehot`: sequence tensor shaped `(N, n_symbols, seq_len)`
- `x_type=continuous`: sequence tensor shaped `(N, seq_len)`
- `label_type=binary`: label tensor shaped `(N, n_attr)`
- `label_type=continuous`: label tensor shaped `(N, 1)` or `(N, n_attr)` depending on the experiment

If `labels_<seq_len>.pth` is missing, [`src/loader.py`](src/loader.py) will generate dummy labels for testing.

## Quick Start

Install dependencies with either:

```bash
pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml
```

Then run a training job, for example:

```bash
python train.py \
  --name lattice_demo \
  --data_path data/lattice_EM/processed \
  --seq_len 27 \
  --alphabet_type lattice \
  --attr binary \
  --encoder_hidden_dims 50,20 \
  --decoder_hidden_dims 50 \
  --dis_hidden_dims 15,10,5 \
  --n_lat_dis 1 \
  --batch_size 64 \
  --n_epochs 10 \
  --cuda False
```

Completed runs are written under `models/<name>/<random_id>/`.

## Utilities and Legacy Files

- [`decoder_weights.py`](decoder_weights.py), [`scripts/full_dataset.py`](scripts/full_dataset.py), [`scripts/sample_sequences.py`](scripts/sample_sequences.py), [`scripts/sample_EM.py`](scripts/sample_EM.py), and [`scripts/print_enc_out.py`](scripts/print_enc_out.py) are checkpoint-inspection helpers.
- [`plot_observables.py`](plot_observables.py) and [`plot_branch_b_results.py`](plot_branch_b_results.py) are plotting utilities for saved experiment outputs.
- [`classifier.py`](classifier.py) and [`interpolate.py`](interpolate.py) are legacy image-domain scripts inherited from the original upstream project. They are not part of the main biological-sequence workflow.

## Tests

The repository includes lightweight smoke tests under [`tests/`](tests). They validate model construction, training-step execution, and sequence loading behavior rather than providing a full regression suite.
