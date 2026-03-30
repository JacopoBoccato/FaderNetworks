#!/usr/bin/env python3
"""
Plot Route B Observables from FaderNetwork Experiments.

This script loads the observable histories saved by run_many_experiments.py
and creates plots showing how each observable evolves across epochs and sweeps.

Usage:
    python plot_observables.py --results_file results/route_b_observables.pkl
"""

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Plot Route B observables from FaderNetwork experiments')
    parser.add_argument('--results_file', type=str, default='results/route_b_observables.pkl',
                       help='Path to the pickled results file from run_many_experiments.py')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='Directory to save plots')
    args = parser.parse_args()
    
    # Load results
    with open(args.results_file, 'rb') as f:
        results = pickle.load(f)
    
    sweep1 = results['sweep1_lambda']
    sweep2 = results['sweep2_eigenvalues']
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Observable names and configurations
    observables = ['rec_loss', 'adv_loss', 'nuisance_overlap', 'signal_capture', 'nuisance_in_code']
    sweep1_labels = [f'λ={lam:.4g}' for lam in sorted(sweep1.keys())]
    sweep2_labels = list(sweep2.keys())
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Sweep 1: λ_lat_dis
    # ─────────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('Sweep 1: λ_lat_dis ∈ {0.001, 0.05, 0.5, 2.0, 10.0}', fontsize=14, fontweight='bold')
    
    for idx, obs_name in enumerate(observables):
        ax = axes[idx]
        for lam in sorted(sweep1.keys()):
            history = sweep1[lam]
            epochs = [h['epoch'] for h in history]
            values = [h[obs_name] for h in history]
            ax.plot(epochs, values, marker='o', label=f'λ={lam:.4g}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(obs_name.replace('_', ' ').title())
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/sweep1_lambda_observables.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {args.output_dir}/sweep1_lambda_observables.png")
    plt.close()
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Sweep 2: Eigenvalues
    # ─────────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('Sweep 2: Eigenvalue Configurations', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(sweep2)))
    for idx, obs_name in enumerate(observables):
        ax = axes[idx]
        for config_idx, (config_name, history) in enumerate(sweep2.items()):
            epochs = [h['epoch'] for h in history]
            values = [h[obs_name] for h in history]
            ax.plot(epochs, values, marker='o', label=config_name, color=colors[config_idx])
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(obs_name.replace('_', ' ').title())
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/sweep2_eigenvalue_observables.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {args.output_dir}/sweep2_eigenvalue_observables.png")
    plt.close()
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Final values comparison across both sweeps
    # ─────────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Final Observable Values After Training', fontsize=14, fontweight='bold')
    
    for idx, obs_name in enumerate(observables):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        
        # Sweep 1
        sweep1_lambdas = sorted(sweep1.keys())
        sweep1_values = [sweep1[lam][-1][obs_name] for lam in sweep1_lambdas]
        x1 = np.arange(len(sweep1_lambdas))
        ax.bar(x1 - 0.2, sweep1_values, 0.4, label='λ sweep', alpha=0.8)
        
        # Sweep 2
        sweep2_configs = list(sweep2.keys())
        sweep2_values = [sweep2[cfg][-1][obs_name] for cfg in sweep2_configs]
        x2 = np.arange(len(sweep2_configs))
        ax.bar(x2 + len(sweep1_lambdas) + 0.2, sweep2_values, 0.4, label='Eigenvalue sweep', alpha=0.8)
        
        ax.set_ylabel(obs_name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3, axis='y')
        
        # X-axis labels
        all_labels = [f'λ={l:.3g}' for l in sweep1_lambdas] + sweep2_configs
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
    
    # Remove the extra subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/final_observable_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {args.output_dir}/final_observable_comparison.png")
    plt.close()
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"OBSERVABLES SUMMARY")
    print(f"{'='*80}")
    print("\nRoute B Observable Definitions:")
    print("  • rec_loss:           ‖X̂ - X‖² (reconstruction error)")
    print("  • adv_loss:           ‖ŷ - y‖² (discriminator prediction error on true y)")
    print("  • nuisance_overlap:   Correlation between latent code z and true nuisance y")
    print("  • signal_capture:     ‖z‖_F (Frobenius norm of latent codes)")
    print("  • nuisance_in_code:   σ²_y × var(z) (nuisance variance leakage)")
    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
