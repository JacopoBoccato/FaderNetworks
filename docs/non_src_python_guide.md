# Non-`src` Python Guide

This guide covers the Python files outside [`src/`](../src), grouped by how you should treat them.

The key distinction is:

- [`src/`](../src): reusable implementation
- everything else: entrypoints, experiments, utilities, and tests

Not every root-level script is equally current. Some are canonical, some are experimental, and some are upstream leftovers.

## 1. Main entrypoints

These are the scripts to read first.

### `train.py`

Role:
- canonical entrypoint for normal sequence Fader training

What it does:
- parses experiment arguments
- loads datasets through `src.loader.load_sequences`
- builds `AutoEncoder`, `LatentDiscriminator`, and optional extra discriminators
- delegates optimization to `src.training.Trainer`

When to use it:
- any time you want to train on sequence data using the repository’s current implementation

### `phase_diagram.py`

Role:
- main strict-linear synthetic theory / training comparison script

What it does:
- generates Gaussian teacher data
- instantiates repository models in a linear regime
- computes Branch-B observables from the learned weights
- integrates the theory ODE
- compares measured trajectories against theory

Why the name is misleading:
- despite the filename, the script currently behaves more like a theory-vs-training comparison than a general phase-diagram runner

## 2. Research scripts with lower trust

These are useful, but they are not the cleanest interfaces in the repository.

### `create_dummy_dataset.py`

Role:
- duplicate synthetic Branch-B experiment script

What it actually does:
- runs a theory-vs-training style experiment similar to `phase_diagram.py`
- builds synthetic Gaussian teacher data
- measures observable trajectories from a strict linear Fader

Why it is lower trust:
- the filename suggests dataset generation, but the file is really another research experiment
- it overlaps heavily with `phase_diagram.py`

## 3. Plotting utilities

These scripts visualize saved experiment outputs rather than train models.

### `plot_observables.py`

Role:
- plotting utility for older pickled observable histories

What it expects:
- a pickle file with sweep results and epoch-wise metrics

Limitation:
- it targets an older saved-results format and is not part of the main training path

### `plot_branch_b_results.py`

Role:
- plotting utility for Branch-B-style sweeps and phase summaries

What it does:
- renders phase diagrams
- renders 1D slices through sweeps
- plots observables such as normalized order parameters and reconstruction error

Limitation:
- it assumes a specific pickled results structure produced by the synthetic research workflow

## 4. Checkpoint inspection utilities

These scripts help inspect trained models after training.

### `decoder_weights.py`

Role:
- inspect the first decoder layer of a saved checkpoint

What it does:
- loads `params.pkl` and `best_rec_ae.pth`
- finds the first decoder linear layer
- measures column norms for latent and attribute inputs
- writes a text summary next to the checkpoint

### `scripts/print_enc_out.py`

Role:
- dump encoder outputs for a saved model

What it does:
- loads a checkpoint and tensors
- runs the encoder
- writes latent vectors and labels to a text file

### `scripts/full_dataset.py`

Role:
- decode a full sequence file and compare original / reconstructed / flipped outputs

What it does:
- loads a saved model
- one-hot encodes raw sequences
- reconstructs them under the original attribute
- decodes them again with flipped attributes
- writes all three sequence strings side by side

### `scripts/sample_sequences.py`

Role:
- deterministic validation sampler for lattice experiments

What it does:
- loads fixed subsets from `LatticeA.txt` and `LatticeB.txt`
- reconstructs and flips them
- computes per-sequence reconstruction losses
- compares the model against a deterministic frequency baseline

### `scripts/sample_EM.py`

Role:
- inspection utility for the processed EM dataset

What it does:
- decodes original and attribute-flipped outputs
- scores decoded sequences with the hand-written charge rule
- writes scores, losses, and decoded strings for manual inspection

## 5. Dataset preparation scripts

These scripts create tensors that can later be used for training.

### `scripts/prepare_EM_dataset.py`

Role:
- build a lattice-style protein dataset with binary labels

What it does:
- reads one sequence per line
- scores each sequence using fixed charge-pattern rules
- keeps only strongly scored examples
- one-hot encodes them with the lattice alphabet
- writes `sequences_<L>.pth` and `labels_<L>.pth`

Why it matters:
- its outputs match the naming convention expected by `train.py`

### `scripts/process_rna_sequences.py`

Role:
- build an RNA dataset from tabular records with taxonomy

What it does:
- parses `ID<TAB>SEQ<TAB>TAXONOMY` records
- normalizes `T -> U`
- one-hot encodes RNA sequences
- creates binary labels based on whether the phylum is `Bacillota`
- writes tensor files and taxonomy metadata

Current caveat:
- it saves `rna_sequences_<L>.pth` and `rna_labels_<L>.pth`, while `train.py` expects `sequences_<L>.pth` and `labels_<L>.pth`

## 6. Legacy image-domain scripts

These are inherited from the original image Fader codebase and are not part of the main sequence workflow.

### `classifier.py`

Role:
- image classifier training entrypoint from the upstream project

Why it is legacy:
- it depends on `src.loader.load_images`
- it expects image-specific parameters such as `img_sz`, `img_fm`, and augmentation flags

### `interpolate.py`

Role:
- image interpolation / attribute-swapping visualizer from the upstream project

Why it is legacy:
- it assumes an image model and image loader path
- it is unrelated to the biological-sequence pipeline

## 7. Tests

These files are support code rather than user-facing scripts.

### `tests/model_test.py`

Role:
- smoke test for `AutoEncoder` construction, forward pass, and backward pass

### `tests/training_test.py`

Role:
- smoke test for one training step across binary and continuous settings

### `tests/test_loader.py`

Role:
- smoke test for alphabet utilities, one-hot encoding, dataset saving, and `DataSampler`

## 8. Shell launchers and cluster jobs

These are not Python files, but they explain how the repository is commonly run.

- `scripts/train_lattice_fader.sh`: single-run shell launcher for a lattice experiment
- `scripts/Fader_rna.sh`: multi-run RNA architecture sweep launcher
- `jobs/*.pbs`: PBS job scripts for cluster execution

## Practical Reading Order

If you are new to the repository, read files in this order:

1. `README.md`
2. `train.py`
3. `src/model.py`
4. `src/training.py`
5. `src/loader.py`
6. the specific utility or experiment script you actually plan to run
