# Quick Reference: Route B Observable System

## TL;DR - Get Started in 3 Commands

```bash
# 1. Run experiments (50 epochs recommended for production)
python run_many_experiments.py --n_epochs 50 --n_samples 5000 --n_valid 500

# 2. Generate plots
python plot_observables.py

# 3. View results
# → plots/sweep1_lambda_observables.png
# → plots/sweep2_eigenvalue_observables.png  
# → plots/final_observable_comparison.png
# → results/route_b_observables.pkl
```

---

## Observable Quick Reference

| Name | What It Measures | Range | Good ← → Bad |
|---|---|---|---|
| **rec_loss** | Reconstruction quality | 0.2-0.8 | Small ← Bigger |
| **adv_loss** | Attribute removal | 0.2-1.0 | Bigger ← Small |
| **nuisance_overlap** | Nuisance capture | 0.0-1.0 | Small ← Bigger |
| **signal_capture** | Signal information | 6-14 | ← Data-dependent |
| **nuisance_in_code** | Nuisance leakage | 0.05-0.15 | Small ← Bigger |

---

## Expected Sweep Behavior

### Sweep 1: Increasing λ_lat_dis (Adversarial Weight)
- **rec_loss** ↑ (slight)
- **adv_loss** ↓ (strong decrease!)
- **nuisance_overlap** ↓ 
- **signal_capture** ≈ stable
- **nuisance_in_code** ↓

### Sweep 2: Varying Eigenvalues
- **rec_loss** ↑ (with condition number)
- **adv_loss** ≈ stable
- **nuisance_overlap** ≈ stable
- **signal_capture** ↑ (with condition number)
- **nuisance_in_code** ↑ (with condition number)

---

## Observable Computation (What Happens Under the Hood)

```python
# At each epoch on validation set:

# 1. Forward pass through FaderNetwork
z = autoencoder.encode(X)        # Latent codes (B, 32)
x_hat = autoencoder.decode(z)    # Reconstruction
y_pred = discriminator(z)         # Attribute prediction

# 2. Compute observables
rec_loss = MSE(x_hat, X)
adv_loss = MSE(y_pred, y)
nuisance_overlap = correlation(z[:, 0], y)  # First latent dimension
signal_capture = frobenius_norm(z)
nuisance_in_code = sigma2_y * variance(z)

# 3. Store in history dict
history.append({
    'epoch': epoch_number,
    'rec_loss': rec_loss_value,
    'adv_loss': adv_loss_value,
    'nuisance_overlap': overlap_value,
    'signal_capture': signal_value,
    'nuisance_in_code': leakage_value
})
```

---

## Loading Results Programmatically

```python
import pickle

# Load results
with open('results/route_b_observables.pkl', 'rb') as f:
    results = pickle.load(f)

# Extract sweep 1 (varying λ_lat_dis)
sweep1 = results['sweep1_lambda']
for lambda_val in sorted(sweep1.keys()):
    history = sweep1[lambda_val]
    print(f"λ={lambda_val}:")
    print(f"  Epochs: {len(history)}")
    print(f"  Final rec_loss: {history[-1]['rec_loss']:.3f}")
    print(f"  Final adv_loss: {history[-1]['adv_loss']:.3f}")

# Extract sweep 2 (varying eigenvalues)
sweep2 = results['sweep2_eigenvalues']
for config_name in sweep2:
    history = sweep2[config_name]
    print(f"{config_name}:")
    print(f"  Signal capture: {history[-1]['signal_capture']:.2f}")

# Access individual epoch (e.g., lambda 0.5, epoch 2)
epoch_data = sweep1[0.5][2]  # Third epoch (index 2)
print(epoch_data)  # → dict with all 5 observables + epoch number
```

---

## Common Tasks

### Compare adversarial weights
```python
# See how each observable changes from λ=0.001 to λ=10.0
start = sweep1[0.001][-1]  # Final epoch, weak adversarial
end = sweep1[10.0][-1]     # Final epoch, strong adversarial

print(f"adv_loss change: {start['adv_loss']:.3f} → {end['adv_loss']:.3f}")
```

### Find which eigenvalue config has lowest reconstruction
```python
configs = sweep2
best_config = min(configs.keys(), 
                  key=lambda k: configs[k][-1]['rec_loss'])
print(f"Best reconstruction: {best_config}")
```

### Plot just one observable across epochs
```python
import matplotlib.pyplot as plt

history = sweep1[0.5]  # λ=0.5 sweep
epochs = [h['epoch'] for h in history]
rec_losses = [h['rec_loss'] for h in history]

plt.plot(epochs, rec_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Reconstruction Loss')
plt.title('Reconstruction Loss Evolution (λ=0.5)')
plt.grid(True, alpha=0.3)
plt.savefig('rec_loss_evolution.png', dpi=150)
```

---

## Files Overview

```
run_many_experiments.py     ← Main experiment runner
  ├─ build_teacher()           Generate synthetic data teacher
  ├─ generate_continuous_dataset()  Create training data
  ├─ compute_observables()      MAP THEORY TO PRACTICE ★
  ├─ train_fader_network()      Train & track observables ★
  └─ main()                     Run sweeps & save results

plot_observables.py         ← Visualization tool
  └─ Creates 3 PNG plots from pickle file

results/
  └─ route_b_observables.pkl  ← Your data (pickled dict)

plots/
  ├─ sweep1_lambda_observables.png
  ├─ sweep2_eigenvalue_observables.png
  └─ final_observable_comparison.png

OBSERVABLES_MAPPING.md      ← Detailed technical docs
IMPLEMENTATION_SUMMARY.md   ← Full implementation guide
```

---

## Troubleshooting

**Q: Plot shows no data?**
- A: Pickle file might be from old test. Re-run: `python run_many_experiments.py --n_epochs 5`

**Q: Want to see per-epoch convergence?**
- A: Increase `--n_epochs` to 20-50 and re-plot

**Q: How to extend with custom observables?**
- A: Edit `compute_observables()` function, add new observable computation in the `with torch.no_grad():` block

**Q: Results look wrong?**
- A: Check that data generation matches Route B theory. See `build_teacher()` and `generate_continuous_dataset()` functions.

---

## Key Commits

| Commit | What Changed |
|---|---|
| 8bd9745 | Observable computation + storage infrastructure |
| 4f0271e | Implementation summary docs |
| HEAD | This quick reference |

---

## Next Steps

1. ✅ Run with full dataset (50 epochs)
2. ✅ Verify observables match Route B theory predictions
3. ✅ Generate publication-quality plots
4. ✅ Analyze which λ_lat_dis value gives best disentanglement
5. ✅ Compare eigenvalue effects on signal capture

---

Generated: Route B Observable System v1.0  
Status: ✅ Ready for production use  
Last updated: 2024
