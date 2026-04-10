# FaderNetworks for Biological Sequences

This repository adapts Fader Networks to biological-sequence style data and also contains a research pipeline for comparing trained linear Fader dynamics against Branch-B theory on synthetic Gaussian teacher data.

It supports two distinct workflows:

1. Standard repository Fader training on sequence-like data using `train.py` and the modules in `src/`.
2. Branch-B theory experiments and phase-diagram sweeps using synthetic data and strict linear versions of the repository models.

## Core Idea

A Fader Network splits training into:

1. An autoencoder that reconstructs `x` from a latent code `z = E(x)` and an attribute `y`.
2. A latent discriminator that tries to predict `y` from `z`.

The decoder receives `y` explicitly, so the encoder is encouraged to remove attribute information from the latent code while preserving the information needed for reconstruction.

In this repository:

- `src/model.py` defines the general `AutoEncoder` and `LatentDiscriminator`.
- `src/training.py` defines the repository training loop and adversarial scheduling.
- `src/utils.py` defines the discriminator lambda ramp through `get_lambda(...)`.

## Repository Structure

For a grouped map of every Python file outside `src/`, including which ones are canonical, legacy, or theory-facing, see [docs/non_src_python_guide.md](docs/non_src_python_guide.md).

### Standard Training

- `train.py`
  Main entrypoint for standard Fader training.

- `src/model.py`
  Core models:
  - `AutoEncoder`
  - `LatentDiscriminator`
  - optional discriminators and loss helpers

- `src/training.py`
  Alternating training logic:
  - latent discriminator step
  - optional other discriminator steps
  - autoencoder step

- `src/utils.py`
  Utilities including optimizer construction and adversarial lambda scheduling.

- `src/loader.py`
  Sequence loading, one-hot / continuous handling, and `DataSampler`.

### Branch-B / Theory Scripts

- `fader_phase_diagrams_repo_linear.py`
  Canonical Branch-B sweep script using the repository models and `Trainer` in strict linear mode.

- `compare_branch_b_dynamics.py`
  Direct theory-vs-training comparison for a single configuration:
  - generates teacher data
  - trains a strict linear Fader
  - integrates the Branch-B ODE
  - compares observables over time
  - saves a separate convergence / loss plot

- `fader_phase_diagrams.py`
  Older standalone phase-diagram implementation. Useful as a reference, but less aligned with repository training semantics than `fader_phase_diagrams_repo_linear.py`.

### Observable / Plotting Utilities

- `plot_observables.py`
  Plots stored observable histories from experiment outputs.

- `plot_branch_b_results.py`
  Plotting utility for Branch-B-style saved grids and observables.

## Installation

The repository expects:

- Python 3
- NumPy
- SciPy
- PyTorch
- Matplotlib
- CUDA if you want GPU training

If you maintain dependencies with `requirements.txt` or `environment.yml`, prefer those over manual installation.

## Standard Fader Loss in This Repository

The repository training logic is defined in `src/training.py`.

For the autoencoder step:

- reconstruction loss is computed with `sequence_reconstruction_loss(...)`
- adversarial pressure is applied through `lambda_lat_dis`
- the adversarial term is ramped with `get_lambda(...)` from `src/utils.py`

For continuous labels, the relevant training form is:

\[
\mathcal{L}_{\mathrm{AE}}
=
\lambda_{\mathrm{ae}} \, \mathcal{L}_{\mathrm{rec}}
+
\lambda_{\mathrm{lat}}(t) \, \mathcal{L}_{\mathrm{lat\_dis}}
\]

where:

- `\mathcal{L}_{rec}` is the reconstruction loss
- `\mathcal{L}_{lat_dis}` is the latent discriminator loss evaluated in the repository convention
- `\lambda_{lat}(t)` is the scheduled adversarial weight from `get_lambda(...)`

The latent discriminator itself is trained in alternating steps to predict the true attribute from the latent code.

Important: the compare / Branch-B scripts may use strict linearized versions of the repository models, but the canonical adversarial scheduling logic should still come from `src/training.py` and `src/utils.py` if you want behavior consistent with the repository.

## Branch-B Teacher Model

The synthetic theory scripts use a Gaussian teacher model of the form:

\[
X = Uc + vy + \sqrt{\eta}\,a
\]

with:

- `U \in R^{n x r}` orthonormal signal subspace
- `v \in R^n` nuisance / attribute direction
- `c` signal latent variable
- `y` scalar continuous attribute
- `a` isotropic noise
- `\eta` reconstruction-noise level

The total noise is split into:

- `eta = ETA_FRACTION * noise_total`
- `g = (1 - ETA_FRACTION) * noise_total`

These values are then reused in both the measured Fader training pipeline and the ODE.

## Main Research Workflows

### 1. Compare Branch-B Theory to Trained Dynamics

Run:

```bash
python compare_branch_b_dynamics.py
```

Useful flags:

```bash
python compare_branch_b_dynamics.py \
  --device cuda \
  --n_epochs 1000 \
  --alpha_C 1 \
  --eta_clf 1.0 \
  --lambda_schedule 50000
```

Outputs:

- `branch_b_theory_vs_training.png`
  Theory vs measured observables.

- `branch_b_training_loss.png`
  Separate training-loss / convergence diagnostic plot.

- `branch_b_theory_vs_training.npz`
  Saved arrays for observables, times, and loss histories.

The script is configured through the `ExperimentConfig` block at the top of `compare_branch_b_dynamics.py`.

### 2. Run Repo-Aligned Branch-B Phase Sweeps

Run:

```bash
python fader_phase_diagrams_repo_linear.py
```

This is the preferred Branch-B sweep script because it:

- uses repository `AutoEncoder` and `LatentDiscriminator`
- uses `Trainer`
- uses the repository lambda schedule
- supports convergence diagnostics
- is closer to the actual repository semantics than the standalone older script

## Observable Definitions Used in the Research Pipeline

The Branch-B scripts track observables derived from the trained linear system:

- `M`, `s`
  Encoder alignment with the signal subspace and nuisance direction.

- `N`, `a`
  Decoder alignment with the signal subspace and nuisance direction.

- `beta`, `rho`
  Decoder bias projections onto signal / nuisance components.

- `Q`, `T`, `u`, `t`, `B`, `m`
  Bulk observables needed for the closed ODE.

Common scalar summaries include:

- `reconstruction_error`
- `norm_M`, `norm_N`, `norm_s`, `norm_a`, ...
- `M_tilde = ||M|| / sqrt(tr(Q))`
- `N_tilde = ||N|| / sqrt(tr(Q))`

These are computed through `src/branch_b_observables.py` and then reduced to plotting-friendly scalar histories inside the scripts.

## Convergence Diagnostics

For theory-facing scripts, the most reliable convergence proxy is usually held-out or full-dataset reconstruction error, not an arbitrary signed saddle objective.

When diagnosing training:

1. Track reconstruction loss across epochs.
2. Track the latent-discriminator loss separately.
3. Use a rolling-window stationarity criterion on the reconstruction curve.

The repository-style discriminator strength is typically ramped by `lambda_schedule`, so you should expect early training and late training to live in different adversarial regimes.

## Recommended Canonical Files

If you want to treat part of the repository as the stable, canonical interface, keep these as the primary entrypoints:

- `train.py`
- `src/`
- `fader_phase_diagrams_repo_linear.py`
- `compare_branch_b_dynamics.py`

Other scripts are better treated as auxiliary, plotting, or legacy research utilities.

## Typical Commands

### Standard Training

```bash
python train.py
```

### Theory Comparison on GPU

```bash
python compare_branch_b_dynamics.py --device cuda
```

### Repo-Linear Sweep

```bash
python fader_phase_diagrams_repo_linear.py
```

### Observable Plotting

```bash
python plot_observables.py
```

## Common Failure Modes

### 1. Loss Curve Looks Wrong

Likely causes:

- plotting a signed adversarial objective instead of a true convergence proxy
- mixing repository loss definitions with custom in-script losses
- forgetting the adversarial lambda schedule
- using too large an adversarial weight too early

### 2. GPU Not Used

Use:

```bash
python compare_branch_b_dynamics.py --device cuda
```

and make sure:

- `torch.cuda.is_available()` is true
- your PyTorch install has CUDA support

### 3. Optimizer String Parsing Fails

The repository optimizer parser expects plain decimal numeric strings, not scientific notation inside optimizer strings. If you build optimizer strings programmatically, format values explicitly.

## References

If you use this code, cite:

[*Fader Networks: Manipulating Images by Sliding Attributes*](https://arxiv.org/pdf/1706.00409.pdf)  
Guillaume Lample, Neil Zeghidour, Nicolas Usunier, Antoine Bordes, Ludovic Denoyer, Marc'Aurelio Ranzato

```bibtex
@inproceedings{lample2017fader,
  title={Fader Networks: Manipulating Images by Sliding Attributes},
  author={Lample, Guillaume and Zeghidour, Neil and Usunier, Nicolas and Bordes, Antoine and Denoyer, Ludovic and Ranzato, Marc'Aurelio},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}
```

Contact: `jacopo.boccato@ipht.fr`
