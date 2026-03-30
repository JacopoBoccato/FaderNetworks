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

def train_fader_network(X_train, y_train, X_valid, y_valid, params):
    """Train a FaderNetwork and return final metrics."""
    # Set up data samplers
    train_data = DataSampler(X_train, y_train, params)
    valid_data = DataSampler(X_valid, y_valid, params)
    
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
    
    # Training loop (simplified)
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
                adv_loss = torch.mean((preds - batch_y)**2)  # MSE for continuous
                loss += params.lambda_lat_dis * adv_loss
            
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
    
    # Evaluate on validation
    ae.eval()
    with torch.no_grad():
        X_valid = X_valid.to(device)
        y_valid = y_valid.to(device)
        enc_outputs, dec_outputs = ae(X_valid, y_valid)
        rec_loss = torch.mean((dec_outputs[-1] - X_valid)**2).item()
        if lat_dis:
            z = enc_outputs[-1]
            preds = lat_dis(z)
            adv_loss = torch.mean((preds - y_valid)**2).item()
        else:
            adv_loss = 0.0
    
    return {'rec_loss': rec_loss, 'adv_loss': adv_loss}


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
    args = parser.parse_args()
    
    # Generate dataset
    print("Generating continuous dataset...")
    X_train, y_train = generate_continuous_dataset(args.n_samples, args.n, args.r, [4.0, 1.0, 0.25], args.sigma2_y, args.eta)
    X_valid, y_valid = generate_continuous_dataset(args.n_valid, args.n, args.r, [4.0, 1.0, 0.25], args.sigma2_y, args.eta)
    
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
    
    # Sweep 1: lambda_lat_dis
    lambda_vals = [0.001, 0.05, 0.5, 2.0, 10.0]
    print(f"\nSweep 1: lambda_lat_dis ∈ {lambda_vals}")
    for lam in lambda_vals:
        params.lambda_lat_dis = lam
        print(f"  lambda_lat_dis={lam:.4g} …", end='', flush=True)
        metrics = train_fader_network(X_train, y_train, X_valid, y_valid, params)
        print(f"  rec={metrics['rec_loss']:.3f}  adv={metrics['adv_loss']:.3f}")
    
    # Sweep 2: Eigenvalues
    eigenvalue_configs = {
        'flat  [1,1,1]': [1.0, 1.0, 1.0],
        'mild  [2,1,0.5]': [2.0, 1.0, 0.5],
        'spread[4,1,0.25]': [4.0, 1.0, 0.25],
        'steep [8,1,0.125]': [8.0, 1.0, 0.125],
        'single[4,0.1,0.01]': [4.0, 0.1, 0.01],
    }
    print(f"\nSweep 2: Λ configs (lambda_lat_dis={args.lambda_lat_dis})")
    for lbl, lam_sig in eigenvalue_configs.items():
        X_train, y_train = generate_continuous_dataset(args.n_samples, args.n, args.r, lam_sig, args.sigma2_y, args.eta)
        X_valid, y_valid = generate_continuous_dataset(args.n_valid, args.n, args.r, lam_sig, args.sigma2_y, args.eta)
        print(f"  {lbl} …", end='', flush=True)
        metrics = train_fader_network(X_train, y_train, X_valid, y_valid, params)
        print(f"  rec={metrics['rec_loss']:.3f}  adv={metrics['adv_loss']:.3f}")


if __name__ == '__main__':
    main()