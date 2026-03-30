# Route B Observables Mapping: Theory to Practice

## Overview

This document explains how the theoretical Route B observables from your PDF calculations are mapped to quantities computable from the actual FaderNetwork training in `run_many_experiments.py`.

## Theoretical Framework

**Teacher Model**: $X = U c + v y + \sqrt{\eta} a$
- $U \in \mathbb{R}^{n \times r}$: Orthonormal signal subspace (columns)
- $v \in \mathbb{R}^n$: Nuisance direction  
- $c, y, a$: i.i.d. standard normal latent variables
- $\eta$: Noise level
- $y$: The continuous attribute (nuisance/undesired signal)

**FaderNetwork Components**:
- AutoEncoder: Maps $X \to z$ (encoder), $z \to \hat{X}$ (decoder)
- LatentDiscriminator: Predicts $y$ from $z$ (adversarial loss)

---

## Observable Definitions & Mappings

### 1. Reconstruction Loss: `rec_loss`

**Theory**: Measures how well the autoencoder reconstructs the data
$$\text{rec\_loss} = \mathbb{E}[\|\hat{X} - X\|_F^2]$$

**Practice**: 
```python
rec_loss = mean((x_hat - X)^2)  # MSE on validation set
```

**Interpretation**: Lower values indicate better reconstruction. This is the primary autoencoder objective.

---

### 2. Adversarial Loss: `adv_loss`

**Theory**: Measures how well the discriminator can predict $y$ from $z$
$$\text{adv\_loss} = \mathbb{E}[\|\hat{y} - y\|^2]$$
where $\hat{y} = \text{LatentDiscriminator}(z)$

**Practice**:
```python
y_pred = lat_dis(z)  # (B, 1)
adv_loss = mean((y_pred - y)^2)  # MSE on validation set
```

**Interpretation**: 
- **Low adv_loss** (< 0.5): Discriminator can easily predict $y$ from $z$ → latent space contains nuisance information
- **High adv_loss** (> 1.0): Discriminator struggles → latent space is disentangled from $y$

---

### 3. Nuisance Overlap: `nuisance_overlap`

**Theory**: How much the encoder captures the nuisance direction $v$
$$s = W @ v \quad \text{(where } W \text{ is encoder weight)}$$
$$\text{nuisance\_overlap} = \frac{|s \cdot C^*|}{|s| |C^*|} \quad \text{(correlation)}$$

**Practice**: 
```python
# Correlation between first latent dimension and true y
z_component = z[:, 0]  # Take first latent code dimension
correlation = abs(corrcoef(z_component, y)[0,1])
nuisance_overlap = correlation
```

**Interpretation**:
- **0-0.3**: Low correlation → encoder not capturing nuisance direction
- **0.5-1.0**: High correlation → encoder strongly capturing nuisance information

---

### 4. Signal Capture: `signal_capture`

**Theory**: Magnitude of encoder projection onto signal subspace
$$\text{signal\_capture} = \|M\|_F \quad \text{where } M = W @ U$$

**Practice**:
```python
# Frobenius norm of latent codes (proxy for encoder output magnitude)
signal_capture = norm(z, 'frobenius')
```

**Interpretation**:
- **6-10**: Typical range indicating encoder captures signal information
- **Invariant to lambda_lat_dis** (should remain stable across adversarial weight sweeps)
- **Varies with eigenvalues Λ**: Larger signal subspace (higher condition number) → larger capture

---

### 5. Nuisance in Code: `nuisance_in_code`

**Theory**: Leakage of nuisance variance into the learned codes
$$\text{nuisance\_in\_code} = \sigma_y^2 \cdot \|s\|^2$$
where $s$ is nuisance projection (as above)

**Practice**:
```python
# Variance of latent codes weighted by true nuisance variance
sigma2_y = 1.0  # Teacher model nuisance variance
nuisance_in_code = sigma2_y * var(z)
```

**Interpretation**:
- **~0.05-0.1**: Well-disentangled latent space (nuisance separated)
- **> 0.15**: Strong nuisance leakage into codes
- **Should decrease** as $\lambda_{\text{lat\_dis}}$ increases (stronger adversarial regularization)

---

## Sweep 1: Varying $\lambda_{\text{lat\_dis}}$

**Fixed**: Data eigenvalues $\Lambda = [4, 1, 0.25]$

**Variable**: Adversarial regularization strength $\lambda_{\text{lat\_dis}} \in \{0.001, 0.05, 0.5, 2.0, 10.0\}$

**Expected Behavior**:
| Observable | Expected Trend | Reason |
|---|---|---|
| rec_loss | ↑ (slight increase) | Regularization constrains encoder |
| adv_loss | ↓ (decreases) | Stronger adversarial penalty |
| nuisance_overlap | ↓ (decreases) | Disentanglement enforced |
| signal_capture | ≈ stable | Signal subspace independent of adversarial weight |
| nuisance_in_code | ↓ (decreases) | Less nuisance variance in codes |

**Interpretation**: As you increase adversarial weight, the model learns a more disentangled representation where nuisance information is removed from the latent space.

---

## Sweep 2: Varying Eigenvalues $\Lambda$

**Fixed**: $\lambda_{\text{lat\_dis}} = 0.5$

**Variable**: Data eigenvalues configurations:
- flat: $[1, 1, 1]$ (isotropic signal)
- mild: $[2, 1, 0.5]$ (weak conditioning)
- spread: $[4, 1, 0.25]$ (moderate conditioning)
- steep: $[8, 1, 0.125]$ (high conditioning)
- single: $[4, 0.1, 0.01]$ (extreme conditioning)

**Expected Behavior**:
| Observable | Expected Trend | Reason |
|---|---|---|
| rec_loss | ↑ with condition number | Hard-to-reconstruct directions |
| adv_loss | ≈ stable | Fixed adversarial strength |
| nuisance_overlap | ≈ stable | Nuisance direction fixed |
| signal_capture | ↑ with condition number | Larger signal subspace norm |
| nuisance_in_code | ↑ with condition number | More variance to capture |

**Interpretation**: The condition number of the signal covariance affects how much information the encoder needs to capture, influencing reconstruction difficulty and code magnitude.

---

## Usage

### 1. Run experiments with observable tracking:
```bash
python run_many_experiments.py \
  --n_epochs 10 \
  --n_samples 1000 \
  --n_valid 100 \
  --out_dir results
```

### 2. Generate plots:
```bash
python plot_observables.py \
  --results_file results/route_b_observables.pkl \
  --output_dir plots
```

### 3. Load and analyze results programmatically:
```python
import pickle
with open('results/route_b_observables.pkl', 'rb') as f:
    results = pickle.load(f)

# Access sweep 1 histories
for lambda_val, history in results['sweep1_lambda'].items():
    print(f"λ={lambda_val}:")
    for epoch_dict in history:
        print(f"  Epoch {epoch_dict['epoch']}: rec={epoch_dict['rec_loss']:.3f}")

# Access sweep 2 histories
for config_name, history in results['sweep2_eigenvalues'].items():
    print(f"{config_name}:")
    final = history[-1]
    print(f"  Final adv_loss={final['adv_loss']:.3f}")
```

---

## Observable Summary Table

| Observable | Units | Scale | Route B Equivalent | Physical Meaning |
|---|---|---|---|---|
| rec_loss | MSE | 0.3-0.8 | $\\|(A W - I) \Sigma_{xx}\\|$ | Reconstruction error |
| adv_loss | MSE | 0.4-1.2 | $Q(W, C^*)$ | Attribute prediction error |
| nuisance_overlap | Correlation | 0.2-0.9 | $\frac{\|s \cdot C^*\|}{\|s\|\|C^*\|}$ | Nuisance capture |
| signal_capture | Norm | 6-10 | $\\|M\\|_F$ | Signal code magnitude |
| nuisance_in_code | Scaled variance | 0.04-0.1 | $\sigma_y^2 \\|s\\|^2$ | Nuisance leakage |

---

## Notes

1. **Dimensionality**: The latent dimension is fixed at 32 (last encoder hidden layer). The observables are computed on the batch-averaged latent codes.

2. **Validation Set**: All observables are computed on the validation set (not training set) to assess generalization.

3. **Numerical Stability**: NaN values in correlation are replaced with 0.0 to handle edge cases (e.g., constant z values).

4. **Teacher Consistency**: The teacher model parameters (U, v, σ²_y) are re-generated for each eigenvalue configuration to match the data generation, ensuring consistent observable computation.

5. **Plotting**: The provided `plot_observables.py` script creates three plots:
   - Epoch-wise evolution for Sweep 1 (λ variations)
   - Epoch-wise evolution for Sweep 2 (eigenvalue variations)
   - Final value comparison across both sweeps

---

## Questions & Troubleshooting

**Q: Why does signal_capture stay relatively constant?**
- A: It's a proxy for encoder output magnitude, which is mainly determined by network initialization and data scale, not much affected by adversarial regularization.

**Q: Why are nuisance_overlap and adv_loss sometimes uncorrelated?**
- A: nuisance_overlap uses only the first latent dimension, while the discriminator uses all dimensions. The discriminator can exploit other dimensions for prediction.

**Q: Should I run longer (more epochs)?**
- A: Yes! Current tests use 2 epochs. For production, use `--n_epochs 50` or more to see observable convergence behavior.

---

Generated: `run_many_experiments.py` v2.0  
Observable computation: `compute_observables()` function  
Visualization: `plot_observables.py`
