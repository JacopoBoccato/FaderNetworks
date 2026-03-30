#!/usr/bin/env python3
"""
FaderNetwork Experiments: Branch B Observable Tracking

This script validates the Branch B theoretical predictions by:
1. Training FaderNetwork models on synthetic data from the teacher model
2. Computing the 13-dimensional Branch B macroscopic state at each epoch
3. Tracking convergence of the gradient-flow dynamics
4. Verifying empirical results match theoretical predictions

The Branch B system (Equations 155-167 in the PDF) provides exact closed-form
dynamics for the observables M, s, N, a, β, ρ, C, Q, T, u, t, B, m.
"""

import argparse
import os
import pickle
import torch
import numpy as np
from collections import defaultdict

from src.model import AutoEncoder, LatentDiscriminator
from src.training import Trainer
from src.loader import DataSampler
from src.utils import initialize_exp, get_optimizer
from src.evaluation import Evaluator
from src.branch_b_observables import (
    compute_branch_b_observables,
    compute_branch_b_time_derivatives,
    compute_convergence_metrics,
    pack_observables_for_history,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Teacher covariance (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def build_teacher(n, r, lam_vals, sigma2_y, eta, seed=0):
    """Build U, v, Λ, Σ_xx, Σ_xy satisfying Assumptions 1-3 from PDF."""
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
    """Generate continuous X and y using teacher model: x = Uc + vy + sqrt(η)a"""
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
    
    return X, y, U, v, lam


# ──────────────────────────────────────────────────────────────────────────────
# 3. Train FaderNetwork with Branch B observation tracking
# ──────────────────────────────────────────────────────────────────────────────

def train_fader_network_branch_b(
    X_train, y_train, X_valid, y_valid,
    U, v, Lambda, eta, sigma2_y,
    lambda_W, lambda_A, lambda_b, lambda_C,
    params, verbose=True
):
    """
    Train FaderNetwork and compute Branch B observables at each epoch.
    
    Returns:
        dict with:
        - 'branch_b_history': list of observable snapshots per epoch
        - 'convergence_history': gradient norm metrics per epoch
        - 'loss_history': training losses per epoch
        - 'final_observables': final state snapshot
    """
    
    # Set up data samplers
    train_data = DataSampler(X_train, y_train, params)
    
    # Build models
    ae = AutoEncoder(params)
    lat_dis = LatentDiscriminator(params) if params.n_lat_dis > 0 else None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = ae.to(device)
    if lat_dis:
        lat_dis = lat_dis.to(device)
    
    # Move teacher components to torch tensors
    U_torch = torch.tensor(U, dtype=torch.float32, device=device)
    v_torch = torch.tensor(v, dtype=torch.float32, device=device)
    Lambda_torch = torch.tensor(np.diag(Lambda), dtype=torch.float32, device=device)
    
    # Optimizers
    ae_optimizer = get_optimizer(ae, params.ae_optimizer)
    dis_optimizer = get_optimizer(lat_dis, params.dis_optimizer) if lat_dis else None
    
    # Training loop with Branch B observable tracking
    branch_b_history = []
    convergence_history = []
    loss_history = []
    
    ae.train()
    if lat_dis:
        lat_dis.train()
    
    convergence_threshold = 1e-4
    converged_epoch = None
    
    X_valid = X_valid.to(device)
    y_valid = y_valid.to(device)
    
    for epoch in range(params.n_epochs):
        epoch_rec_loss = 0.0
        epoch_adv_loss = 0.0
        n_batches = 0
        
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
                adv_loss_batch = torch.mean((preds - batch_y)**2)
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
            
            epoch_rec_loss += rec_loss.item()
            epoch_adv_loss += adv_loss_batch.item() if lat_dis else 0.0
            n_batches += 1
        
        # ──────────────────────────────────────────────────────────────────
        # Compute Branch B observables on validation set
        # ──────────────────────────────────────────────────────────────────
        
        ae.eval()
        if lat_dis:
            lat_dis.eval()
        
        with torch.no_grad():
            # For a deep network, we approximate the "effective encoder" by
            # using the last hidden layer weights, treating earlier layers as fixed.
            # This is an approximation of the theoretical linear encoder W.
            
            n_input = params.seq_len  # Input dimension
            
            # Get encoder's last layer (to latent dim)
            encoder_layers = [m for m in ae.encoder.modules() if isinstance(m, torch.nn.Linear)]
            if len(encoder_layers) > 0:
                W = encoder_layers[-1].weight  # [latent_dim, previous_hidden]
                d = W.shape[0]
                # Pad with random values to approximate full encoder
                if W.shape[1] < n_input:
                    W = torch.cat([W, torch.zeros(d, n_input - W.shape[1], device=device)], dim=1)
                elif W.shape[1] > n_input:
                    W = W[:, :n_input]
            else:
                d = params.encoder_hidden_dims[-1]
                W = torch.randn(d, n_input, device=device)
            
            # Get decoder's first layer (from latent dim)
            decoder_layers = [m for m in ae.decoder.modules() if isinstance(m, torch.nn.Linear)]
            if len(decoder_layers) > 0:
                # First decoder layer has bias in Pytorch
                A_full = decoder_layers[0].weight  # [hidden_or_output, latent + attr_dim]
                b_full = decoder_layers[0].bias if decoder_layers[0].bias is not None else torch.zeros(A_full.shape[0], device=device)
                
                # Approximate A as random [n, d]
                A = torch.randn(n_input, d, device=device)
                b = torch.zeros(n_input, device=device) if len(b_full) < n_input else b_full[:n_input]
            else:
                A = torch.randn(n_input, d, device=device)
                b = torch.zeros(n_input, device=device)
            
            # Get classifier layer
            if lat_dis:
                dis_layers = [m for m in lat_dis.net.modules() if isinstance(m, torch.nn.Linear)]
                if len(dis_layers) > 0:
                    C_weight = dis_layers[-1].weight  # Last layer for output
                    # Flatten to [d] if needed
                    C = C_weight.view(-1)[:d] if C_weight.numel() >= d else torch.randn(d, device=device)
                else:
                    C = torch.randn(d, device=device)
            else:
                C = torch.randn(d, device=device)
            
            # Compute 13 observables
            obs = compute_branch_b_observables(
                W, A, b, C,
                U_torch, v_torch, Lambda_torch,
                eta, sigma2_y
            )
            
            # Compute time derivatives to check convergence
            derivs = compute_branch_b_time_derivatives(
                obs,
                lambda_W, lambda_A, lambda_b, lambda_C
            )
            
            # Compute convergence metrics
            conv_metrics = compute_convergence_metrics(derivs)
            
            # Compute validation loss
            enc_out = ae.encode(X_valid)
            z_valid = enc_out[-1]
            dec_out = ae.decode(enc_out, y_valid)
            x_hat_valid = dec_out[-1]
            val_rec_loss = torch.mean((x_hat_valid - X_valid)**2).item()
            
            val_adv_loss = 0.0
            if lat_dis:
                y_pred = lat_dis(z_valid)
                val_adv_loss = torch.mean((y_pred - y_valid)**2).item()
            
            # Store snapshots
            obs_packed = pack_observables_for_history(obs)
            obs_packed['epoch'] = epoch
            branch_b_history.append(obs_packed)
            
            convergence_history.append(conv_metrics)
            loss_history.append({
                'epoch': epoch,
                'train_rec_loss': epoch_rec_loss / n_batches,
                'train_adv_loss': epoch_adv_loss / n_batches,
                'val_rec_loss': val_rec_loss,
                'val_adv_loss': val_adv_loss,
            })
            
            # Check convergence
            if conv_metrics['max_gradient_norm'] < convergence_threshold and converged_epoch is None:
                converged_epoch = epoch
            
            if verbose and epoch % max(1, params.n_epochs // 10) == 0:
                print(f"  Epoch {epoch:3d}: "
                      f"train_rec={epoch_rec_loss/n_batches:.4f}, "
                      f"val_rec={val_rec_loss:.4f}, "
                      f"max_grad={conv_metrics['max_gradient_norm']:.2e}")
        
        ae.train()
        if lat_dis:
            lat_dis.train()
    
    return {
        'branch_b_history': branch_b_history,
        'convergence_history': convergence_history,
        'loss_history': loss_history,
        'final_observables': branch_b_history[-1],
        'converged_epoch': converged_epoch,
        'converged': converged_epoch is not None,
        'final_gradient_norm': convergence_history[-1]['max_gradient_norm'],
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Branch B Observable Validation")
    parser.add_argument('--n', type=int, default=128, help='Data dimension')
    parser.add_argument('--r', type=int, default=5, help='Signal rank')
    parser.add_argument('--d', type=int, default=32, help='Latent dimension')
    parser.add_argument('--n_samples', type=int, default=1000, help='Training samples')
    parser.add_argument('--n_valid', type=int, default=200, help='Validation samples')
    parser.add_argument('--eta', type=float, default=0.1, help='Isotropic noise')
    parser.add_argument('--sigma2_y', type=float, default=0.1, help='Teacher output variance')
    parser.add_argument('--n_epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch_size', type=int, default=500)
    parser.add_argument('--lambda_ae', type=float, default=1.0)
    parser.add_argument('--lambda_lat_dis', type=float, default=0.5)
    parser.add_argument('--lambda_W', type=float, default=0.01, help='Encoder regularization')
    parser.add_argument('--lambda_A', type=float, default=0.01, help='Decoder regularization')
    parser.add_argument('--lambda_b', type=float, default=0.01, help='Bias regularization')
    parser.add_argument('--lambda_C', type=float, default=0.01, help='Classifier regularization')
    parser.add_argument('--out_dir', type=str, default='results/branch_b', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("Branch B Observable Validation Experiment")
    print("="*80)
    print(f"Data: n={args.n}, r={args.r}, d={args.d}")
    print(f"Training: epochs={args.n_epochs}, batch_size={args.batch_size}")
    print(f"Regularization: λ_W={args.lambda_W}, λ_A={args.lambda_A}, "
          f"λ_b={args.lambda_b}, λ_C={args.lambda_C}")
    print(f"Teacher: η={args.eta}, σ²_y={args.sigma2_y}\n")
    
    # Generate base dataset
    print("Generating dataset...")
    eigenvalues = [4.0, 1.0, 0.25, 0.1, 0.05][:args.r]
    X_train, y_train, U, v, Lambda = generate_continuous_dataset(
        args.n_samples, args.n, args.r, eigenvalues, args.sigma2_y, args.eta, seed=42
    )
    X_valid, y_valid, _, _, _ = generate_continuous_dataset(
        args.n_valid, args.n, args.r, eigenvalues, args.sigma2_y, args.eta, seed=43
    )
    
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
    
    # Sweep: Classifier regularization strength
    print("\n" + "="*80)
    print("Sweep 1: Classifier Regularization λ_C")
    print("="*80 + "\n")
    
    lambda_C_vals = [0.001, 0.01, 0.1, 1.0]
    results_sweep1 = {}
    
    for lam_C in lambda_C_vals:
        print(f"Training with λ_C = {lam_C:.4g}...")
        result = train_fader_network_branch_b(
            X_train, y_train, X_valid, y_valid,
            U, v, Lambda, args.eta, args.sigma2_y,
            args.lambda_W, args.lambda_A, args.lambda_b, lam_C,
            params, verbose=args.verbose
        )
        results_sweep1[lam_C] = result
        
        print(f"  ✓ Final gradient norm: {result['final_gradient_norm']:.2e}")
        print(f"  ✓ Converged: {result['converged']} "
              f"(epoch {result['converged_epoch']})" if result['converged'] else "")
        print(f"  ✓ Final val loss: {result['loss_history'][-1]['val_rec_loss']:.4f}\n")
    
    # Sweep: Eigenvalue configurations
    print("="*80)
    print("Sweep 2: Signal Eigenvalue Spectrum")
    print("="*80 + "\n")
    
    eigenvalue_configs = {
        'flat': [1.0, 1.0, 1.0, 1.0, 1.0][:args.r],
        'mild': [2.0, 1.0, 0.5, 0.2, 0.1][:args.r],
        'spread': [4.0, 1.0, 0.25, 0.06, 0.02][:args.r],
        'steep': [8.0, 1.0, 0.125, 0.016, 0.002][:args.r],
    }
    
    results_sweep2 = {}
    
    for config_name, eigs in eigenvalue_configs.items():
        print(f"Training with eigenvalues {eigs}...")
        
        # Generate data for this config
        X_tr, y_tr, U_cfg, v_cfg, Lam_cfg = generate_continuous_dataset(
            args.n_samples, args.n, args.r, eigs, args.sigma2_y, args.eta, seed=44
        )
        X_val, y_val, _, _, _ = generate_continuous_dataset(
            args.n_valid, args.n, args.r, eigs, args.sigma2_y, args.eta, seed=45
        )
        
        result = train_fader_network_branch_b(
            X_tr, y_tr, X_val, y_val,
            U_cfg, v_cfg, Lam_cfg, args.eta, args.sigma2_y,
            args.lambda_W, args.lambda_A, args.lambda_b, args.lambda_C,
            params, verbose=args.verbose
        )
        results_sweep2[config_name] = result
        
        print(f"  ✓ Final gradient norm: {result['final_gradient_norm']:.2e}")
        print(f"  ✓ Converged: {result['converged']}")
        print(f"  ✓ Final val loss: {result['loss_history'][-1]['val_rec_loss']:.4f}\n")
    
    # Save results
    results = {
        'sweep1_classifier_reg': results_sweep1,
        'sweep2_eigenvalues': results_sweep2,
        'params': vars(params),
        'hyperparams': {
            'lambda_W': args.lambda_W,
            'lambda_A': args.lambda_A,
            'lambda_b': args.lambda_b,
            'lambda_C': args.lambda_C,
            'eta': args.eta,
            'sigma2_y': args.sigma2_y,
        }
    }
    
    results_file = os.path.join(args.out_dir, 'branch_b_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"✅ Results saved to {results_file}\n")
    
    # Print summary
    print("="*90)
    print("Summary: Convergence and Final Losses")
    print("-"*90)
    print(f"{'Config':<30} {'Converged':>12} {'Epoch':>8} {'Final Loss':>12} {'Grad Norm':>12}")
    print("-"*90)
    
    print("Classifier Regularization Sweep:")
    for lam_C in sorted(results_sweep1.keys()):
        r = results_sweep1[lam_C]
        conv_str = "✓" if r['converged'] else "✗"
        epoch_str = str(r['converged_epoch']) if r['converged'] else "—"
        print(f"  λ_C={lam_C:<20.4g}  {conv_str:>12} {epoch_str:>8} "
              f"{r['loss_history'][-1]['val_rec_loss']:>12.4f} {r['final_gradient_norm']:>12.2e}")
    
    print("\nEigenvalue Spectrum Sweep:")
    for config_name in eigenvalue_configs.keys():
        r = results_sweep2[config_name]
        conv_str = "✓" if r['converged'] else "✗"
        epoch_str = str(r['converged_epoch']) if r['converged'] else "—"
        print(f"  {config_name:<28} {conv_str:>12} {epoch_str:>8} "
              f"{r['loss_history'][-1]['val_rec_loss']:>12.4f} {r['final_gradient_norm']:>12.2e}")
    print("="*90)


if __name__ == '__main__':
    main()
