#!/usr/bin/env python3
"""
Branch B Observable Visualization

Load and plot Branch B observable histories from pickle files.
Generates publication-quality figures showing:
1. Observable evolution over epochs
2. Convergence verification (gradient norms)
3. Loss trajectories
4. Parameter trajectories
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_branch_b_observables(results_file, output_dir='plots/branch_b'):
    """
    Plot all Branch B observable histories.
    
    Args:
        results_file: Path to branch_b_results.pkl
        output_dir: Where to save PNG files
    """
    
    # Load results
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot Sweep 1: Classifier Regularization
    plot_classifier_reg_sweep(
        results['sweep1_classifier_reg'],
        output_dir / 'sweep1_classifier_regularization.png'
    )
    
    # Plot Sweep 2: Eigenvalue Spectrum
    plot_eigenvalue_sweep(
        results['sweep2_eigenvalues'],
        output_dir / 'sweep2_eigenvalue_spectrum.png'
    )
    
    # Plot convergence comparison
    plot_convergence_comparison(
        results['sweep1_classifier_reg'],
        results['sweep2_eigenvalues'],
        output_dir / 'convergence_comparison.png'
    )
    
    print(f"✅ Plots saved to {output_dir}")


def plot_classifier_reg_sweep(sweep_results, save_path):
    """Plot observable evolution for classifier regularization sweep."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Sweep 1: Classifier Regularization λ_C', fontsize=14, fontweight='bold')
    
    ax_loss = axes[0, 0]
    ax_grad = axes[0, 1]
    ax_m_norm = axes[0, 2]
    ax_s_norm = axes[1, 0]
    ax_q_norm = axes[1, 1]
    ax_b_norm = axes[1, 2]
    
    for lam_C, result in sorted(sweep_results.items()):
        epochs = np.arange(len(result['loss_history']))
        
        # Validation loss
        val_losses = [h['val_rec_loss'] for h in result['loss_history']]
        ax_loss.semilogy(epochs, val_losses, marker='o', label=f'λ_C={lam_C}')
        
        # Gradient norm (convergence)
        grad_norms = [h['max_gradient_norm'] for h in result['convergence_history']]
        ax_grad.semilogy(epochs, grad_norms, marker='s', label=f'λ_C={lam_C}')
        
        # Observable norms
        m_norms = [h['norm_M'] for h in result['branch_b_history']]
        s_norms = [h['norm_s'] for h in result['branch_b_history']]
        q_norms = [h['norm_Q'] for h in result['branch_b_history']]
        b_norms = [h['sqrt_m'] for h in result['branch_b_history']]
        
        ax_m_norm.plot(epochs, m_norms, marker='o', label=f'λ_C={lam_C}')
        ax_s_norm.plot(epochs, s_norms, marker='s', label=f'λ_C={lam_C}')
        ax_q_norm.plot(epochs, q_norms, marker='^', label=f'λ_C={lam_C}')
        ax_b_norm.plot(epochs, b_norms, marker='d', label=f'λ_C={lam_C}')
    
    # Formatting
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Validation Loss')
    ax_loss.set_title('Reconstruction Loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    
    ax_grad.set_xlabel('Epoch')
    ax_grad.set_ylabel('Max Gradient Norm')
    ax_grad.set_title('Convergence (Lower = Better)')
    ax_grad.axhline(1e-4, color='red', linestyle='--', alpha=0.5, label='Convergence threshold')
    ax_grad.legend()
    ax_grad.grid(True, alpha=0.3)
    
    ax_m_norm.set_xlabel('Epoch')
    ax_m_norm.set_ylabel('‖M‖_F')
    ax_m_norm.set_title('Encoder-Signal Projection')
    ax_m_norm.legend()
    ax_m_norm.grid(True, alpha=0.3)
    
    ax_s_norm.set_xlabel('Epoch')
    ax_s_norm.set_ylabel('‖s‖')
    ax_s_norm.set_title('Encoder-Nuisance Projection')
    ax_s_norm.legend()
    ax_s_norm.grid(True, alpha=0.3)
    
    ax_q_norm.set_xlabel('Epoch')
    ax_q_norm.set_ylabel('‖Q‖_F')
    ax_q_norm.set_title('Encoder Autocorrelation')
    ax_q_norm.legend()
    ax_q_norm.grid(True, alpha=0.3)
    
    ax_b_norm.set_xlabel('Epoch')
    ax_b_norm.set_ylabel('√m')
    ax_b_norm.set_title('Bias Norm')
    ax_b_norm.legend()
    ax_b_norm.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


def plot_eigenvalue_sweep(sweep_results, save_path):
    """Plot observable evolution for eigenvalue spectrum sweep."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Sweep 2: Signal Eigenvalue Spectrum', fontsize=14, fontweight='bold')
    
    ax_loss = axes[0, 0]
    ax_grad = axes[0, 1]
    ax_m_norm = axes[0, 2]
    ax_n_norm = axes[1, 0]
    ax_t_norm = axes[1, 1]
    ax_b_norm = axes[1, 2]
    
    for config_name, result in sweep_results.items():
        epochs = np.arange(len(result['loss_history']))
        
        # Validation loss
        val_losses = [h['val_rec_loss'] for h in result['loss_history']]
        ax_loss.semilogy(epochs, val_losses, marker='o', label=config_name)
        
        # Gradient norm (convergence)
        grad_norms = [h['max_gradient_norm'] for h in result['convergence_history']]
        ax_grad.semilogy(epochs, grad_norms, marker='s', label=config_name)
        
        # Observable norms
        m_norms = [h['norm_M'] for h in result['branch_b_history']]
        n_norms = [h['norm_N'] for h in result['branch_b_history']]
        t_norms = [h['norm_T'] for h in result['branch_b_history']]
        b_norms = [h['sqrt_m'] for h in result['branch_b_history']]
        
        ax_m_norm.plot(epochs, m_norms, marker='o', label=config_name)
        ax_n_norm.plot(epochs, n_norms, marker='s', label=config_name)
        ax_t_norm.plot(epochs, t_norms, marker='^', label=config_name)
        ax_b_norm.plot(epochs, b_norms, marker='d', label=config_name)
    
    # Formatting
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Validation Loss')
    ax_loss.set_title('Reconstruction Loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    
    ax_grad.set_xlabel('Epoch')
    ax_grad.set_ylabel('Max Gradient Norm')
    ax_grad.set_title('Convergence (Lower = Better)')
    ax_grad.axhline(1e-4, color='red', linestyle='--', alpha=0.5, label='Convergence threshold')
    ax_grad.legend()
    ax_grad.grid(True, alpha=0.3)
    
    ax_m_norm.set_xlabel('Epoch')
    ax_m_norm.set_ylabel('‖M‖_F')
    ax_m_norm.set_title('Encoder-Signal Projection')
    ax_m_norm.legend()
    ax_m_norm.grid(True, alpha=0.3)
    
    ax_n_norm.set_xlabel('Epoch')
    ax_n_norm.set_ylabel('‖N‖_F')
    ax_n_norm.set_title('Decoder-Signal Projection')
    ax_n_norm.legend()
    ax_n_norm.grid(True, alpha=0.3)
    
    ax_t_norm.set_xlabel('Epoch')
    ax_t_norm.set_ylabel('‖T‖_F')
    ax_t_norm.set_title('Decoder Gram Matrix')
    ax_t_norm.legend()
    ax_t_norm.grid(True, alpha=0.3)
    
    ax_b_norm.set_xlabel('Epoch')
    ax_b_norm.set_ylabel('√m')
    ax_b_norm.set_title('Bias Norm')
    ax_b_norm.legend()
    ax_b_norm.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


def plot_convergence_comparison(sweep1, sweep2, save_path):
    """Plot convergence across both sweeps."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sweep 1: Classifier regularization
    for lam_C, result in sorted(sweep1.items()):
        epochs = np.arange(len(result['convergence_history']))
        grad_norms = [h['max_gradient_norm'] for h in result['convergence_history']]
        ax1.semilogy(epochs, grad_norms, marker='o', label=f'λ_C={lam_C}')
    
    ax1.axhline(1e-4, color='red', linestyle='--', linewidth=2, label='Conv threshold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Max Gradient Norm (log scale)', fontsize=12)
    ax1.set_title('Sweep 1: Classifier Regularization', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sweep 2: Eigenvalue spectrum
    for config_name, result in sweep2.items():
        epochs = np.arange(len(result['convergence_history']))
        grad_norms = [h['max_gradient_norm'] for h in result['convergence_history']]
        ax2.semilogy(epochs, grad_norms, marker='s', label=config_name)
    
    ax2.axhline(1e-4, color='red', linestyle='--', linewidth=2, label='Conv threshold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Max Gradient Norm (log scale)', fontsize=12)
    ax2.set_title('Sweep 2: Eigenvalue Spectrum', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle('Branch B Convergence Verification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot Branch B Observable Evolution')
    parser.add_argument('--results', type=str, default='results/branch_b/branch_b_results.pkl',
                       help='Path to pickle file with results')
    parser.add_argument('--output', type=str, default='plots/branch_b',
                       help='Output directory for PNG files')
    args = parser.parse_args()
    
    plot_branch_b_observables(args.results, args.output)
