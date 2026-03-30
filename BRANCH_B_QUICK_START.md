# Branch B Theory Implementation - Quick Start Guide

## TL;DR: The 13 Observable State Variables

Branch B theory defines a **13-dimensional closed system** that governs FaderNetwork training:

| Projected Observables | Shape | Meaning |
|---|---|---|
| $M = WU$ | $d \times r$ | Encoder projects onto signal subspace |
| $s = Wv$ | $d$ | Encoder projects onto nuisance direction |
| $N = U^T A$ | $r \times d$ | Decoder weights in signal frame |
| $a = A^T v$ | $d$ | Decoder aligns with nuisance |
| $\beta = U^T b$ | $r$ | Bias component in signal space |
| $\rho = v^T b$ | scalar | Bias component in nuisance direction |
| $C$ | $d$ | Classifier vector |

| Bulk Observables | Shape | Meaning |
|---|---|---|
| $Q = WW^T$ | $d \times d$ | Encoder covariance |
| $T = A^T A$ | $d \times d$ | Decoder Gram matrix |
| $u = A^T b$ | $d$ | Decoder-bias coupling |
| $t = Wb$ | $d$ | Encoder-bias coupling |
| $B = A^T W^T$ | $d \times d$ | Decoder-encoder correlation |
| $m = b^T b$ | scalar | Bias norm squared |

## How to Use in `run_many_experiments.py`

### 1. Import and Extract Observables

```python
from src.branch_b_observables import (
    compute_branch_b_observables,
    compute_branch_b_time_derivatives,
    compute_convergence_metrics,
    pack_observables_for_history
)

# After training each epoch:
W = model.encoder.weight  # [d, n]
A = model.decoder_A.weight  # [n, d]
b = model.decoder_bias  # [n]
C = discriminator.classifier.weight.squeeze()  # [d]

# Compute all 13 observables
obs = compute_branch_b_observables(
    W, A, b, C,
    U, v, Lambda, eta, sigma2_y
)

# Get time derivatives to verify convergence
derivs = compute_branch_b_time_derivatives(obs, lambda_W, lambda_A, lambda_b, lambda_C)

# Check convergence
conv_metrics = compute_convergence_metrics(derivs)
print(f"Max gradient norm: {conv_metrics['max_gradient_norm']:.6e}")

# Store in history
history['branch_b'].append(pack_observables_for_history(obs))
history['convergence'].append(conv_metrics)
```

### 2. Interpret Convergence

- **Converged**: $\max_i |\dot{x}_i| < 10^{-4}$ or $10^{-5}$
- **Not converged**: $\max_i |\dot{x}_i| > 10^{-3}$

If gradients plateau but don't reach zero:
- May indicate incorrect teacher covariance specification
- Or learning rate settings not matching theory assumptions

### 3. Plot Results

Create 5 subplots showing:

1. **Parameter Evolution**: Norms of M, N, Q, T, B over epochs
2. **Reconstruction Error**: Loss curves for both terms
3. **Convergence Check**: Log-scale gradient norms
4. **Bias Components**: Evolution of β, ρ, √m
5. **Adversarial Terms**: Contributions to classifier loss

## Example Integration

```python
# In run_many_experiments.py training loop:

for epoch in range(n_epochs):
    # ... training code ...
    
    # Extract theory quantities
    with torch.no_grad():
        obs = compute_branch_b_observables(
            model.encoder.weight,
            model.decoder_A.weight,
            model.decoder_bias,
            discriminator.classifier.weight.squeeze(),
            U, v, Lambda, eta, sigma2_y
        )
        
        derivs = compute_branch_b_time_derivatives(
            obs, lambda_W, lambda_A, lambda_b, lambda_C
        )
        
        conv = compute_convergence_metrics(derivs)
        
        # Check if converged
        if conv['max_gradient_norm'] < 1e-4:
            print(f"Converged at epoch {epoch}")
            break
        
        # Store history
        obs_dict = pack_observables_for_history(obs)
        for key, val in obs_dict.items():
            history[f'branch_b/{key}'].append(val)
        for key, val in conv.items():
            history[f'convergence/{key}'].append(val)

# Plot results
plot_branch_b_results(history, save_path='plots/branch_b.png')
```

## Theory vs. Practice Matching

For exact match with theory:
1. ✓ Check teacher covariance: $\Sigma_{xx} = U\Lambda U^T + \sigma_y^2 vv^T + \eta I_n$
2. ✓ Verify data generation: $x = Uc + vy + \sqrt{\eta}a$
3. ✓ Confirm initialization: parameters should start near zero
4. ✓ Use continuous-time gradient flow assumption
5. ✓ Verify batch size matches "infinite population" regime

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| Diverging gradients | Wrong teacher covariance | Check U, v, Λ construction |
| Oscillating loss | Learning rate too high | Reduce LR or use smaller time steps |
| No convergence | Rank deficiency in N or M | Check signal rank r matches setup |
| Non-smooth evolution | Discrete batch effects | Use large batch size or epoch averaging |

## Files

- **`BRANCH_B_IMPLEMENTATION_GUIDE.md`**: Detailed equations and mapping
- **`src/branch_b_observables.py`**: Core computation functions
- **`BRANCH_B_QUICK_START.md`**: This file
- **`run_many_experiments.py`**: Modified to use Branch B tracking (TODO)
