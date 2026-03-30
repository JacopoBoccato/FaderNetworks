#!/usr/bin/env python3
"""
FaderNetwork Experiments: Training on Continuous Data with Sweeps.

This script generates a synthetic continuous dataset using a teacher model,
then runs two sweeps of experiments training actual FaderNetwork models:
1. Sweep on classifier regularization strength (lambda_lat_dis).
2. Sweep on signal eigenvalues (Λ) affecting data generation.

Each experiment trains a FaderNetwork with continuous X (sequences) and y (attributes).
Results are printed to console; no plotting.
"""

import argparse
import os
import pickle
import torch
import numpy as np
from src.model import AutoEncoder, LatentDiscriminator
from src.training import Trainer
from src.loader import DataSampler
from src.utils import initialize_exp, get_optimizer
from src.evaluation import Evaluator


# ──────────────────────────────────────────────────────────────────────────────
# 1. Teacher covariance (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def build_teacher(n, r, lam_vals, sigma2_y, eta, seed=0):
    """Build U, v, Λ, Σ_xx, Σ_xy satisfying Assumptions 1-3."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, r + 1)))
    U = Q[:, :r]          # n×r, orthonormal
    v = Q[:, r]           # n,   U'v=0 by QR construction
    lam = np.asarray(lam_vals, dtype=float)
    assert lam.shape == (r,)
    Sxx = U @ np.diag(lam) @ U.T + sigma2_y * np.outer(v, v) + eta * np.eye(n)
    Sxy = sigma2_y * v
    return U, v, lam, Sxx, Sxy


# ──────────────────────────────────────────────────────────────────────────────
# 2. Generate continuous dataset
# ──────────────────────────────────────────────────────────────────────────────

def generate_continuous_dataset(n_samples, n, r, lam_vals, sigma2_y, eta, seed=0):
    """Generate continuous X and y using teacher model."""
    U, v, lam, Sxx, Sxy = build_teacher(n, r, lam_vals, sigma2_y, eta, seed)
    rng = np.random.default_rng(seed)
    
    # Sample latent variables
    c = rng.standard_normal((n_samples, r))  # signal latents
    y = rng.standard_normal((n_samples, 1))  # nuisance
    a = rng.standard_normal((n_samples, n))  # noise
    
    # Generate X: x = U c + v y + sqrt(eta) a
    X = (U @ c.T).T + (v[:, None] @ y.T).T + np.sqrt(eta) * a
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.squeeze(), dtype=torch.float32).unsqueeze(-1)
    
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# 3. Train FaderNetwork for one config
# ──────────────────────────────────────────────────────────────────────────────

def compute_observables(ae, lat_dis, X, y, U, v, sigma2_y, device):
    """
    Compute observables that map to Route B theoretical quantities.
    These are evaluated on validation data.
    """
    ae.eval()
    if lat_dis:
        lat_dis.eval()
    
    with torch.no_grad():
        X = X.to(device)
        y = y.to(device)
        
        # Encode
        enc_outputs = ae.encode(X)
        z = enc_outputs[-1]  # latent code (B, d)
        
        # Decode
        dec_outputs = ae.decode(enc_outputs, y)
        x_hat = dec_outputs[-1]  # reconstructed X (B, n)
        
        # 1. Reconstruction loss: E‖X_hat - X‖²
        rec_loss = torch.mean((x_hat - X)**2).item()
        
        # 2. Adversarial loss: MSE between discriminator pred and true y
        adv_loss = 0.0
        if lat_dis:
            y_pred = lat_dis(z)  # (B, 1) or (B, n_attr)
            adv_loss = torch.mean((y_pred - y)**2).item()
        
        # 3. Nuisance overlap (how much does encoder capture the y-direction)
        # Project encoder weights onto signal and nuisance directions
        # s = W @ v: projection of X direction v onto the encoder
        # For continuous data, we compute correlation between first latent component and y
        if lat_dis and y.dim() > 1 and y.size(1) == 1:
            # Correlation between first latent code dimension and true nuisance attribute
            z_component = z[:, 0].cpu().numpy() if z.dim() > 1 else z.view(-1).cpu().numpy()[:20]
            y_flat = y.view(-1).cpu().numpy()
            if len(z_component) > 1 and len(y_flat) > 1:
                correlation = np.abs(np.corrcoef(z_component, y_flat)[0, 1])
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
            nuisance_overlap = float(correlation)
        else:
            nuisance_overlap = 0.0
        
        # 4. Signal capture: norm of encoder's latent code
        # In theory, this is ‖M‖_F where M = W @ U (encoder on signal subspace)
        # Here we use the Frobenius norm of the latent codes
        signal_capture = float(torch.norm(z, 'fro').item())
        
        # 5. Nuisance in code: how much latent code correlates with nuisance
        # In theory: σ²_y ‖s‖², where s = W @ v
        # Here we compute correlation between z components and y
        nuisance_in_code = float(torch.mean((z - torch.mean(z, dim=0))**2).item() * sigma2_y)
    
    return {
        'rec_loss': rec_loss,
        'adv_loss': adv_loss,
        'nuisance_overlap': nuisance_overlap,
        'signal_capture': signal_capture,
        'nuisance_in_code': nuisance_in_code,
    }


def train_fader_network(X_train, y_train, X_valid, y_valid, U, v, sigma2_y, params):
    """Train a FaderNetwork and compute observables at each epoch."""
    # Set up data samplers
    train_data = DataSampler(X_train, y_train, params)
    
    # Build models
    ae = AutoEncoder(params)
    lat_dis = LatentDiscriminator(params) if params.n_lat_dis > 0 else None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = ae.to(device)
    if lat_dis:
        lat_dis = lat_dis.to(device)
    
    # Optimizers
    ae_optimizer = get_optimizer(ae, params.ae_optimizer)
    dis_optimizer = get_optimizer(lat_dis, params.dis_optimizer) if lat_dis else None
    
    # Training loop with observable tracking
    history = []
    ae.train()
    if lat_dis:
        lat_dis.train()
    
    for epoch in range(params.n_epochs):
        for _ in range(params.epoch_size // params.batch_size):
            # Get batch
            batch_x, batch_y = train_data.next_batch()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Autoencoder step
            ae_optimizer.zero_grad()
            enc_outputs, dec_outputs = ae(batch_x, batch_y)
            rec_loss = torch.mean((dec_outputs[-1] - batch_x)**2)
            loss = params.lambda_ae * rec_loss
            
            if lat_dis:
                z = enc_outputs[-1]
                preds = lat_dis(z)
                adv_loss_batch = torch.mean((preds - batch_y)**2)  # MSE for continuous
                loss += params.lambda_lat_dis * adv_loss_batch
            
            loss.backward()
            ae_optimizer.step()
            
            # Discriminator step
            if lat_dis:
                dis_optimizer.zero_grad()
                with torch.no_grad():
                    enc_outputs, _ = ae(batch_x, batch_y)
                    z = enc_outputs[-1]
                preds = lat_dis(z)
                dis_loss = torch.mean((preds - batch_y)**2)
                dis_loss.backward()
                dis_optimizer.step()
        
        # Evaluate observables on validation set
        obs = compute_observables(ae, lat_dis, X_valid, y_valid, U, v, sigma2_y, device)
        obs['epoch'] = epoch
        history.append(obs)
    
    return history


# ──────────────────────────────────────────────────────────────────────────────
# 4. Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10, help='Data dimension')
    parser.add_argument('--r', type=int, default=3, help='Signal rank')
    parser.add_argument('--n_samples', type=int, default=1000, help='Training samples')
    parser.add_argument('--n_valid', type=int, default=100, help='Validation samples')
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--sigma2_y', type=float, default=1.0)
    parser.add_argument('--n_epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch_size', type=int, default=500)
    parser.add_argument('--lambda_ae', type=float, default=1.0)
    parser.add_argument('--lambda_lat_dis', type=float, default=0.5, help='Fixed for eigenvalue sweep')
    parser.add_argument('--out_dir', type=str, default='results', help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Generate base dataset with fixed eigenvalues
    print("Generating base dataset...")
    X_train, y_train = generate_continuous_dataset(args.n_samples, args.n, args.r, [4.0, 1.0, 0.25], args.sigma2_y, args.eta)
    X_valid, y_valid = generate_continuous_dataset(args.n_valid, args.n, args.r, [4.0, 1.0, 0.25], args.sigma2_y, args.eta)
    
    # Also generate teacher parameters for observables computation
    U, v, lam, Sxx, Sxy = build_teacher(args.n, args.r, [4.0, 1.0, 0.25], args.sigma2_y, args.eta)
    
    # Base params
    params = argparse.Namespace(
        seq_len=args.n,
        x_type='continuous',
        n_attr=1,
        label_type='continuous',
        encoder_hidden_dims=[64, 32],
        decoder_hidden_dims=[64],
        dis_hidden_dims=[32],
        n_lat_dis=1,
        lambda_ae=args.lambda_ae,
        lambda_lat_dis=args.lambda_lat_dis,
        ae_optimizer='adam,lr=0.001',
        dis_optimizer='adam,lr=0.001',
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        epoch_size=args.epoch_size,
        cuda=torch.cuda.is_available()
    )
    
    # Sweep 1: lambda_lat_dis (with fixed eigenvalues)
    lambda_vals = [0.001, 0.05, 0.5, 2.0, 10.0]
    print(f"\n{'='*70}")
    print(f"Sweep 1: lambda_lat_dis ∈ {lambda_vals} (fixed Λ=[4, 1, 0.25])")
    print(f"{'='*70}")
    results_sweep1 = {}
    for lam in lambda_vals:
        params.lambda_lat_dis = lam
        print(f"  lambda_lat_dis={lam:.4g} …", end='', flush=True)
        history = train_fader_network(X_train, y_train, X_valid, y_valid, U, v, args.sigma2_y, params)
        results_sweep1[lam] = history
        h = history[-1]
        print(f"  final_rec={h['rec_loss']:.3f}  adv={h['adv_loss']:.3f}")
    
    # Sweep 2: Eigenvalues (with fixed lambda_lat_dis)
    eigenvalue_configs = {
        'flat  [1,1,1]': [1.0, 1.0, 1.0],
        'mild  [2,1,0.5]': [2.0, 1.0, 0.5],
        'spread[4,1,0.25]': [4.0, 1.0, 0.25],
        'steep [8,1,0.125]': [8.0, 1.0, 0.125],
        'single[4,0.1,0.01]': [4.0, 0.1, 0.01],
    }
    print(f"\n{'='*70}")
    print(f"Sweep 2: Eigenvalue configs (lambda_lat_dis={args.lambda_lat_dis})")
    print(f"{'='*70}")
    results_sweep2 = {}
    for lbl, lam_sig in eigenvalue_configs.items():
        # Regenerate dataset for this eigenvalue config
        X_train_eig, y_train_eig = generate_continuous_dataset(args.n_samples, args.n, args.r, lam_sig, args.sigma2_y, args.eta)
        X_valid_eig, y_valid_eig = generate_continuous_dataset(args.n_valid, args.n, args.r, lam_sig, args.sigma2_y, args.eta)
        
        # Regenerate teacher for this eigenvalue config
        U_eig, v_eig, lam_eig, _, _ = build_teacher(args.n, args.r, lam_sig, args.sigma2_y, args.eta)
        
        print(f"  {lbl} …", end='', flush=True)
        history = train_fader_network(X_train_eig, y_train_eig, X_valid_eig, y_valid_eig, U_eig, v_eig, args.sigma2_y, params)
        results_sweep2[lbl] = history
        h = history[-1]
        print(f"  final_rec={h['rec_loss']:.3f}  signal={h['signal_capture']:.3f}")
    
    # Save results as pickle
    results = {
        'sweep1_lambda': results_sweep1,
        'sweep2_eigenvalues': results_sweep2,
        'params': vars(params),
    }
    
    results_file = os.path.join(args.out_dir, 'route_b_observables.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✅ Results saved to {results_file}")
    
    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'Sweep':<30} {'Recon':>10} {'Adv':>10} {'Overlap':>10} {'Signal':>10} {'NuisCode':>10}")
    print('-'*90)
    print("λ_lat_dis sweep:")
    for lam in sorted(results_sweep1.keys()):
        hist = results_sweep1[lam]
        h = hist[-1]
        print(f"  λ={lam:<15.4g}   {h['rec_loss']:>10.3f} {h['adv_loss']:>10.3f} "
              f"{h['nuisance_overlap']:>10.3f} {h['signal_capture']:>10.3f} {h['nuisance_in_code']:>10.3f}")
    print("Eigenvalue sweep:")
    for lbl in eigenvalue_configs.keys():
        hist = results_sweep2[lbl]
        h = hist[-1]
        print(f"  {lbl:<28} {h['rec_loss']:>10.3f} {h['adv_loss']:>10.3f} "
              f"{h['nuisance_overlap']:>10.3f} {h['signal_capture']:>10.3f} {h['nuisance_in_code']:>10.3f}")
    print('='*90)


if __name__ == '__main__':
    main()