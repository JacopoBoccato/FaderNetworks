# Non-`src` Python Guide

This repository has two layers:

- `src/`: reusable implementation code.
- everything outside `src/`: entrypoints, experiments, utilities, dataset prep, and tests.

The root of the repository is therefore not a flat set of equally important scripts. A better mental model is to group the non-`src` files by role and trust level.

## 1. Stable entrypoints

These are the scripts to treat as the main interfaces of the repo.

### `train.py`

Role:
- Main training entrypoint for the sequence Fader implementation.

What it does:
- parses experiment arguments
- infers alphabet defaults
- loads data through `src.loader`
- builds `AutoEncoder`, `LatentDiscriminator`, optional extra discriminators
- delegates optimization to `src.training.Trainer`

Why it matters:
- this is the canonical path for normal repository training on sequence-like data

### `compare_branch_b_dynamics.py`

Role:
- Main theory-to-training comparison script.

What it does:
- generates synthetic Gaussian teacher data
- instantiates the repository `AutoEncoder` and `LatentDiscriminator`
- forces them into a strict linear regime
- maps learned weights into Branch-B observables
- integrates the closed ODE on those observables
- compares measured training dynamics against theory

Why it matters:
- this is the clearest answer to the question “does the repo-trained linear Fader follow the Branch-B ODE?”

### `fader_phase_diagrams_repo_linear.py`

Role:
- Repo-aligned 2D phase sweep for the same strict-linear Branch-B setting.

What it does:
- sweeps two control parameters
- trains one strict linear Fader per grid point
- extracts observables using `src.branch_b_observables`
- classifies the final phase and plots the phase diagram

Why it matters:
- this is the multi-configuration companion to `compare_branch_b_dynamics.py`

## 2. Branch-B research scripts with lower trust

These are useful, but they are less cleanly aligned with repository semantics.

### `fader_phase_diagrams.py`

Role:
- older standalone linear Fader sweep

What it does:
- defines its own `LinearFader` class instead of reusing `src.model.AutoEncoder`
- trains with hand-written optimization logic
- computes Branch-B observables and phase labels

Why it is lower trust:
- it is closer to a standalone research prototype than to the repository training path

### `run_many_experiments_branch_b.py`

Role:
- older observable-tracking experiment runner

What it does:
- trains on synthetic teacher data
- logs Branch-B observables over epochs
- stores convergence-style histories

Main limitation:
- it uses approximations to recover effective weights from deeper repository models, so it is not as clean a theory match as the strict-linear comparison script

### `run_many_experiments.py`

Role:
- earlier sweep script for continuous-data Fader experiments

What it does:
- runs simple synthetic sweeps
- computes coarse observables such as reconstruction loss, adversarial loss, nuisance overlap, and signal capture

Main limitation:
- the observables are heuristic summaries, not the full closed Branch-B state

### `create_dummy_dataset.py`

Role:
- misleadingly named experimental duplicate of the Branch-B comparison pipeline

What it actually contains:
- a large copy of the same theory-vs-training machinery used in `compare_branch_b_dynamics.py`

Why it is confusing:
- the filename suggests dataset generation, but the file is really another Branch-B comparison script
- if you are choosing one file to read, prefer `compare_branch_b_dynamics.py`

## 3. Plotting and reporting utilities

These scripts consume saved outputs and make figures. They are downstream of the training and sweep scripts.

### `plot_branch_b_results.py`

Role:
- plotting utility for Branch-B-style grids and slices

What it does:
- renders 2D phase diagrams
- renders 1D slices through a sweep
- plots multiple observables with publication-oriented styling

Typical input:
- precomputed sweep arrays stored in `.npz` outputs

### `plot_observables.py`

Role:
- plotting utility for the older `run_many_experiments.py` output format

What it does:
- reads pickled result histories
- plots epoch-wise curves for heuristic observables
- compares final values across two sweeps

Main limitation:
- it is tied to the older heuristic observable pipeline, not the full Branch-B state

## 4. Model inspection and checkpoint utilities

These scripts inspect trained checkpoints rather than train new models.

### `decoder_weights.py`

Role:
- decoder-weight inspection helper

What it does:
- loads a checkpoint
- finds the first decoder linear layer
- measures column norms
- compares latent columns against attribute columns

Use case:
- quick sanity check for whether the decoder is relying strongly on explicit attributes

### `scripts/print_enc_out.py`

Role:
- dumps encoder outputs for a trained checkpoint

What it does:
- loads a saved model and tensors
- runs the encoder
- writes latent vectors with labels to a text file

### `scripts/full_dataset.py`

Role:
- full-dataset decode-and-flip inspection utility

What it does:
- loads a saved model
- reconstructs original sequences
- decodes again with flipped attributes
- writes original, reconstructed, and flipped sequences side by side

### `scripts/sample_sequences.py`

Role:
- sequence-level validation sampler for lattice data

What it does:
- reconstructs and flips a fixed sample
- computes per-sequence reconstruction loss
- compares against a deterministic dummy baseline

### `scripts/sample_EM.py`

Role:
- sampled decode-and-score utility for the EM dataset

What it does:
- decodes original and flipped outputs
- scores decoded sequences with the hand-written charge-pattern rule
- writes scores and reconstructions for inspection

## 5. Dataset preparation scripts

These scripts create tensors or labels from raw sequence data.

### `scripts/process_rna_sequences.py`

Role:
- RNA preprocessing pipeline

What it does:
- parses tabular records with taxonomy
- normalizes RNA sequences
- one-hot encodes them
- creates binary labels from the phylum field
- saves tensors and taxonomy metadata

### `scripts/prepare_EM_dataset.py`

Role:
- protein dataset builder for the EM charge-pattern task

What it does:
- scores sequences using fixed position-specific charge rules
- filters for strong positive or negative patterns
- creates binary labels
- saves one-hot tensors and labels

## 6. Legacy image-only scripts

These are inherited from the original Fader codebase and do not belong to the current sequence/Branch-B core.

### `classifier.py`

Role:
- original image-domain classifier training entrypoint

What it depends on:
- `src.loader.load_images`
- image-specific parameters such as `img_sz`, `img_fm`, `h_flip`

Interpretation:
- useful only if you still care about the original image Fader workflow

### `interpolate.py`

Role:
- original image interpolation / attribute-swapping visualization script

What it depends on:
- torchvision grids
- image loader path
- a single boolean attribute setup

Interpretation:
- legacy, not part of the biological-sequence or Branch-B story

## 7. Tests

These are outside `src/`, but they are support code rather than user-facing scripts.

### `tests/model_test.py`
- unit-style checks for model behavior

### `tests/training_test.py`
- training-loop coverage

### `tests/test_loader.py`
- loader-related checks

## 8. Recommended reading order

If you want the shortest path to understanding the repo, read files in this order:

1. `src/model.py`
2. `src/training.py`
3. `train.py`
4. `src/branch_b_observables.py`
5. `compare_branch_b_dynamics.py`
6. `fader_phase_diagrams_repo_linear.py`

## 9. Focus: how `compare_branch_b_dynamics.py` matches Fader dynamics to the ODE

This script is the key bridge between the microscopic repository implementation and the macroscopic Branch-B theory.

### Step 1: build one shared experimental world

`ExperimentConfig` is the single source of truth for:

- teacher-data parameters
- Fader training parameters
- theory integration horizon
- scheduler and learning-rate settings

`build_shared_params(...)` converts that config into the theory variables:

- `lam_sig`: signal strength
- `eta`: reconstruction-noise contribution
- `g`: label/noise contribution
- `lambda_reg`: weight decay used for encoder/decoder/bias
- `lambda_C`: classifier regularization
- `eta_clf`: adversarial/classifier coupling
- `h_vec`: overlap between nuisance direction and signal subspace

The important design choice is that the same parameter object feeds both the training experiment and the ODE.

### Step 2: generate teacher data that matches the theory assumptions

`build_teacher(...)` constructs:

- `U`: orthonormal signal subspace
- `v`: nuisance direction with controlled overlap `U^T v = h`

`generate_dataset(...)` then samples:

- latent signal coefficients `c`
- nuisance scalar `y`
- isotropic noise `a`

and forms

`X = U c_scaled + v y + sqrt(eta) a`

with isotropic signal covariance `lam_sig * I`.

This matters because the Branch-B ODE is written in terms of these teacher objects.

### Step 3: force the repository model into the exact linear structure required by the theory

`make_linear_fader(...)` uses the real repository classes:

- `src.model.AutoEncoder`
- `src.model.LatentDiscriminator`

but chooses layer dimensions so that each component collapses to one linear map:

- encoder: `z = W x`
- decoder: `x_hat = A z + b y`
- discriminator/classifier: `y_hat = C z`

It also zeroes all biases and freezes them, so the learned system matches the linear theory more closely.

This is the most important structural trick in the file.

### Step 4: keep training semantics aligned with the repo

`build_trainer_params(...)` creates a `SimpleNamespace` that looks like normal repository training parameters.

That namespace is then passed into:

- `DataSampler`
- `Trainer`
- `get_lambda(...)`

So the script does not invent a separate training loop. It reuses:

- the repo optimizer parsing
- the alternating discriminator/AE steps
- the repository lambda ramp schedule
- the continuous-label loss convention from `get_attr_loss(...)`

This is why the script is more trustworthy than the older prototypes.

### Step 5: extract microscopic weights and project them to macroscopic observables

`extract_linear_matrices(...)` reads the trained linear layers and returns:

- `W`: encoder matrix
- `A`: decoder matrix for latent variables
- `b`: decoder column attached to the explicit label input
- `C`: latent discriminator weight vector

`compute_measured_observables(...)` passes those together with `U`, `v`, `Lambda`, `eta`, and `g` into `src.branch_b_observables.compute_branch_b_observables(...)`.

That function computes the full Branch-B state:

- projected observables: `M`, `s`, `N`, `a`, `beta`, `rho`, `C`
- bulk observables: `Q`, `T`, `u`, `t`, `B`, `m`

This step is the actual “matching” between neural-network parameters and ODE state variables.

### Step 6: convert the observable state into an ODE state vector

The theory solver works on a flat vector.

The script therefore defines:

- `pack_state(...)`
- `unpack_state(...)`
- `observables_to_state(...)`

These are pure bookkeeping helpers. They make it possible to:

- compute the initial ODE state from the neural network at initialization
- run the ODE with `solve_ivp(...)`
- convert the solved trajectories back into scalar summaries for plotting

### Step 7: integrate the Branch-B ODE using the same control parameters

`rhs(...)` is the closed macroscopic dynamics.

It evolves all Branch-B variables:

- `M`, `s`, `N`, `a`, `beta`, `rho`, `C`
- `Q`, `T`, `u`, `t`, `B`, `m`

using the same teacher parameters and regularization strengths that the training loop uses.

Two implementation details matter:

- the classifier coupling enters through `classifier_strength(tau, params)`, which mimics the repository lambda schedule in continuous time
- the right-hand side includes regularization for encoder, decoder, bias, and classifier weights explicitly

### Step 8: match training time to theory time

The repository optimizer is discrete-time SGD, while the ODE is continuous-time gradient flow.

The script matches them by defining one effective `dt` per joint update:

- one discriminator update block
- one autoencoder update

`matched_dt = theory_t_final / total_joint_updates`

This is not a theorem. It is a pragmatic calibration so that:

- the measured trajectory
- the ODE trajectory

share the same time horizon and are comparable on one plot.

The script also keeps a second clock, `sgd_times`, based on cumulative learning rate, but the main comparison uses the matched time horizon.

### Step 9: measure the network trajectory repeatedly during training

`train_and_measure(...)` alternates real repository training steps and observable measurements.

At each measurement point it stores scalarized summaries such as:

- `||M||_F`
- `||N||_F`
- `||s||`
- `||a||`
- `||beta||`
- `rho`
- `||Q||_F`
- `||T||_F`
- `||u||`
- `||t||`
- `||B||_F`
- `m`
- reconstruction error
- normalized order parameters `M_tilde`, `N_tilde`

The theoretical trajectory is scalarized in the same way, so the final comparison is apples-to-apples at the observable level.

### Step 10: compare theory and training, not weights and weights

The script never tries to compare raw neural-network parameters directly against ODE coordinates.

Instead it compares:

- theory trajectory in observable space
- measured training trajectory in the same observable space

This is the right comparison, because the Branch-B theory is closed in observables, not in the raw matrices themselves.

## 10. Practical conclusions

If you want to understand the Branch-B story in this repo:

- trust `phase_diagram.py`