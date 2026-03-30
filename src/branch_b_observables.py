"""
Branch B Observable Computation and Convergence Tracking

This module implements the 13-dimensional Branch B macroscopic state
as defined in the PDF (Section 4.2, Equations 155-167).
"""

import torch
import numpy as np
from typing import Dict, Tuple, List
import math


def compute_branch_b_observables(
    W: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    U: torch.Tensor,
    v: torch.Tensor,
    Lambda: torch.Tensor,
    eta: float,
    sigma2_y: float,
) -> Dict[str, torch.Tensor]:
    """
    Compute all 13 Branch B observable state variables.
    
    Args:
        W: Encoder weight matrix [d, n]
        A: Decoder weight matrix [n, d]
        b: Decoder bias vector [n]
        C: Classifier vector [d]
        U: Teacher signal subspace [n, r]
        v: Teacher nuisance direction [n]
        Lambda: Eigenvalue matrix diag(λ1, ..., λr) [r, r]
        eta: Isotropic noise level (scalar)
        sigma2_y: Teacher output variance (scalar)
    
    Returns:
        Dictionary with all 13 observables and auxiliary quantities:
        - Projected observables: M, s, N, a, β, ρ, C
        - Bulk observables: Q, T, u, t, B, m
        - Auxiliary: S, G, J, H, q
    """
    
    device = W.device
    d, n = W.shape
    _, r = U.shape
    
    # Ensure all inputs are on same device
    A = A.to(device)
    b = b.to(device)
    C = C.to(device)
    U = U.to(device)
    v = v.to(device)
    Lambda = Lambda.to(device)
    
    # ============ Projected Observables ============
    
    # (84): M = W U ∈ ℝ^(d×r)
    M = torch.mm(W, U)  # [d, r]
    
    # (84): s = W v ∈ ℝ^d
    s = torch.mv(W, v)  # [d]
    
    # (84): N = U^T A ∈ ℝ^(r×d)
    N = torch.mm(U.t(), A)  # [r, d]
    
    # (84): a = A^T v ∈ ℝ^d
    a = torch.mv(A.t(), v)  # [d]
    
    # (85): β = U^T b ∈ ℝ^r
    beta = torch.mv(U.t(), b)  # [r]
    
    # (85): ρ = v^T b ∈ ℝ (scalar)
    rho = torch.dot(v, b)  # scalar
    
    # ============ Bulk Observables ============
    
    # (85): Q = W W^T ∈ ℝ^(d×d)
    Q = torch.mm(W, W.t())  # [d, d]
    
    # (85): T = A^T A ∈ ℝ^(d×d)
    T = torch.mm(A.t(), A)  # [d, d]
    
    # (85): u = A^T b ∈ ℝ^d
    u = torch.mv(A.t(), b)  # [d]
    
    # (85): t = W b ∈ ℝ^d
    t = torch.mv(W, b)  # [d]
    
    # (85): B = A^T W^T ∈ ℝ^(d×d)
    B = torch.mm(A.t(), W.t())  # [d, d]
    
    # (85): m = b^T b ∈ ℝ (scalar)
    m = torch.dot(b, b)  # scalar
    
    # ============ Auxiliary Definitions ============
    
    g = sigma2_y
    D = Lambda + eta * torch.eye(r, device=device)  # [r, r]
    kappa = g + eta  # scalar
    
    # (150): S = M Λ M^T + g ss^T + ηQ ∈ ℝ^(d×d)
    S = torch.mm(M, torch.mm(Lambda, M.t())) + g * torch.outer(s, s) + eta * Q
    
    # (151): G = N^T Λ M^T + g as^T + ηB ∈ ℝ^(d×d)
    G = torch.mm(N.t(), torch.mm(Lambda, M.t())) + g * torch.outer(a, s) + eta * B
    
    # (152): J = N^T Λ N + g aa^T + ηT ∈ ℝ^(d×d)
    J = torch.mm(N.t(), torch.mm(Lambda, N)) + g * torch.outer(a, a) + eta * T
    
    # (153): H = M Λ β + gρs + ηt ∈ ℝ^d
    H = torch.mv(M, torch.mv(Lambda, beta)) + g * rho * s + eta * t
    
    # (154): q = N^T Λ β + gρa + ηu ∈ ℝ^d
    q = torch.mv(N.t(), torch.mv(Lambda, beta)) + g * rho * a + eta * u
    
    return {
        # Projected observables
        'M': M,
        's': s,
        'N': N,
        'a': a,
        'beta': beta,
        'rho': rho,
        'C': C,
        # Bulk observables
        'Q': Q,
        'T': T,
        'u': u,
        't': t,
        'B': B,
        'm': m,
        # Auxiliary quantities
        'S': S,
        'G': G,
        'J': J,
        'H': H,
        'q': q,
        # Parameters
        'D': D,
        'kappa': kappa,
        'g': g,
        'eta': eta,
    }


def compute_branch_b_time_derivatives(
    observables: Dict[str, torch.Tensor],
    lambda_W: float,
    lambda_A: float,
    lambda_b: float,
    lambda_C: float,
) -> Dict[str, torch.Tensor]:
    """
    Compute time derivatives for all 13 Branch B state variables.
    Implements Equations 155-167 from the PDF.
    
    Args:
        observables: Dictionary from compute_branch_b_observables()
        lambda_W, lambda_A, lambda_b, lambda_C: Regularization parameters
    
    Returns:
        Dictionary with time derivatives: dM/dt, ds/dt, ..., dm/dt
    """
    
    # Extract observables
    M = observables['M']
    s = observables['s']
    N = observables['N']
    a = observables['a']
    beta = observables['beta']
    rho = observables['rho']
    C = observables['C']
    Q = observables['Q']
    T = observables['T']
    u = observables['u']
    t = observables['t']
    B = observables['B']
    m = observables['m']
    S = observables['S']
    G = observables['G']
    J = observables['J']
    H = observables['H']
    q = observables['q']
    D = observables['D']
    kappa = observables['kappa']
    g = observables['g']
    
    d = M.shape[0]
    device = M.device
    
    # Common term
    CC_T = torch.outer(C, C)  # [d, d]
    
    # ============ Equation 155: dM/dt ============
    # Ṁ = -2(TMD - N^T D) + 2CC^T MD - 2λ_W M
    dM_dt = (-2 * (torch.mm(T, torch.mm(M, D)) - torch.mm(N.t(), D))
             + 2 * torch.mm(CC_T, torch.mm(M, D))
             - 2 * lambda_W * M)
    
    # ============ Equation 156: ds/dt ============
    # ṡ = -2(κT s - κa + gu) + 2κCC^T s - 2gC - 2λ_W s
    ds_dt = (-2 * (kappa * torch.mv(T, s) - kappa * a + g * u)
             + 2 * kappa * torch.mv(CC_T, s)
             - 2 * g * C
             - 2 * lambda_W * s)
    
    # ============ Equation 157: dN/dt ============
    # Ṅ = -2(NS - DM^T + gβs^T) - 2λ_A N
    dN_dt = (-2 * (torch.mm(N, S) - torch.mm(D, M.t()) + g * torch.outer(beta, s))
             - 2 * lambda_A * N)
    
    # ============ Equation 158: da/dt ============
    # ȧ = -2(Sa - κs + gρs) - 2λ_A a
    da_dt = (-2 * (torch.mv(S, a) - kappa * s + g * rho * s)
             - 2 * lambda_A * a)
    
    # ============ Equation 159: dβ/dt ============
    # β̇ = -2g(Ns + β) - 2λ_b β
    dbeta_dt = (-2 * g * (torch.mv(N, s) + beta)
                - 2 * lambda_b * beta)
    
    # ============ Equation 160: dρ/dt ============
    # ρ̇ = -2g(a^T s - 1 + ρ) - 2λ_b ρ
    drho_dt = (-2 * g * (torch.dot(a, s) - 1 + rho)
               - 2 * lambda_b * rho)
    
    # ============ Equation 161: dC/dt ============
    # Ċ = -2(SC - gs) - 2λ_C C
    dC_dt = (-2 * (torch.mv(S, C) - g * s)
             - 2 * lambda_C * C)
    
    # ============ Equation 162: dQ/dt ============
    # Q̇ = -2(TS - G + gus^T) - 2(TS - G + gus^T)^T + 2CC^T S + 2SCC^T 
    #     - 2g(Cs^T + sC^T) - 4λ_W Q
    TS_G_term = torch.mm(T, S) - G + g * torch.outer(u, s)
    dQ_dt = (-2 * TS_G_term
             - 2 * TS_G_term.t()
             + 2 * torch.mm(CC_T, S)
             + 2 * torch.mm(S, CC_T)
             - 2 * g * (torch.outer(C, s) + torch.outer(s, C))
             - 4 * lambda_W * Q)
    
    # ============ Equation 163: dT/dt ============
    # Ṫ = -2(TS - G + gus^T) - 2(TS - G + gus^T)^T - 4λ_A T
    dT_dt = (-2 * TS_G_term
             - 2 * TS_G_term.t()
             - 4 * lambda_A * T)
    
    # ============ Equation 164: du/dt ============
    # u̇ = -2(Su - H + gms) - 2g(Ts - a + u) - 2(λ_A + λ_b)u
    du_dt = (-2 * (torch.mv(S, u) - H + g * m * s)
             - 2 * g * (torch.mv(T, s) - a + u)
             - 2 * (lambda_A + lambda_b) * u)
    
    # ============ Equation 165: dt/dt ============
    # ṫ = -2(TH - q + gρu) + 2CC^T H - 2gρC - 2g(B^T s - s + t) - 2(λ_W + λ_b)t
    dt_dt = (-2 * (torch.mv(T, H) - q + g * rho * u)
             + 2 * torch.mv(CC_T, H)
             - 2 * g * rho * C
             - 2 * g * (torch.mv(B.t(), s) - s + t)
             - 2 * (lambda_W + lambda_b) * t)
    
    # ============ Equation 166: dB/dt ============
    # Ḃ = -2(SB - S + gst^T) - 2(GT - J + gau^T) + 2GCC^T - 2gaC^T - 2(λ_A + λ_W)B
    dB_dt = (-2 * (torch.mm(S, B) - S + g * torch.outer(s, t))
             - 2 * (torch.mm(G, T) - J + g * torch.outer(a, u))
             + 2 * torch.mm(G, CC_T)
             - 2 * g * torch.outer(a, C)
             - 2 * (lambda_A + lambda_W) * B)
    
    # ============ Equation 167: dm/dt ============
    # ṁ = -4g(u^T s - ρ + m) - 4λ_b m
    dm_dt = (-4 * g * (torch.dot(u, s) - rho + m)
             - 4 * lambda_b * m)
    
    return {
        'dM_dt': dM_dt,
        'ds_dt': ds_dt,
        'dN_dt': dN_dt,
        'da_dt': da_dt,
        'dbeta_dt': dbeta_dt,
        'drho_dt': drho_dt,
        'dC_dt': dC_dt,
        'dQ_dt': dQ_dt,
        'dT_dt': dT_dt,
        'du_dt': du_dt,
        'dt_dt': dt_dt,
        'dB_dt': dB_dt,
        'dm_dt': dm_dt,
    }


def compute_convergence_metrics(
    time_derivatives: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute convergence metrics from time derivatives.
    
    Convergence is verified when all gradients approach zero.
    
    Args:
        time_derivatives: Dictionary from compute_branch_b_time_derivatives()
    
    Returns:
        Dictionary with norms of each time derivative
    """
    
    metrics = {
        'norm_dM_dt': torch.norm(time_derivatives['dM_dt']).item(),
        'norm_ds_dt': torch.norm(time_derivatives['ds_dt']).item(),
        'norm_dN_dt': torch.norm(time_derivatives['dN_dt']).item(),
        'norm_da_dt': torch.norm(time_derivatives['da_dt']).item(),
        'norm_dbeta_dt': torch.norm(time_derivatives['dbeta_dt']).item(),
        'norm_drho_dt': abs(time_derivatives['drho_dt'].item()),
        'norm_dC_dt': torch.norm(time_derivatives['dC_dt']).item(),
        'norm_dQ_dt': torch.norm(time_derivatives['dQ_dt']).item(),
        'norm_dT_dt': torch.norm(time_derivatives['dT_dt']).item(),
        'norm_du_dt': torch.norm(time_derivatives['du_dt']).item(),
        'norm_dt_dt': torch.norm(time_derivatives['dt_dt']).item(),
        'norm_dB_dt': torch.norm(time_derivatives['dB_dt']).item(),
        'norm_dm_dt': abs(time_derivatives['dm_dt'].item()),
    }
    
    # Overall convergence: maximum gradient norm
    metrics['max_gradient_norm'] = max(metrics.values())
    metrics['mean_gradient_norm'] = np.mean(list(metrics.values()))
    
    return metrics


def frobenius_norm(tensor: torch.Tensor) -> float:
    """Compute Frobenius norm of a matrix."""
    return torch.norm(tensor, p='fro').item()


def pack_observables_for_history(observables: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Convert observable tensors to scalars for history storage.
    
    Only stores norms/scalars to keep memory usage manageable.
    """
    
    return {
        # Norms of projected observables
        'norm_M': frobenius_norm(observables['M']),
        'norm_s': torch.norm(observables['s']).item(),
        'norm_N': frobenius_norm(observables['N']),
        'norm_a': torch.norm(observables['a']).item(),
        'norm_beta': torch.norm(observables['beta']).item(),
        'abs_rho': abs(observables['rho'].item()),
        'norm_C': torch.norm(observables['C']).item(),
        # Norms of bulk observables
        'norm_Q': frobenius_norm(observables['Q']),
        'norm_T': frobenius_norm(observables['T']),
        'norm_u': torch.norm(observables['u']).item(),
        'norm_t': torch.norm(observables['t']).item(),
        'norm_B': frobenius_norm(observables['B']),
        'sqrt_m': math.sqrt(observables['m'].item()),
        # Auxiliary
        'norm_S': frobenius_norm(observables['S']),
        'norm_G': frobenius_norm(observables['G']),
        'norm_J': frobenius_norm(observables['J']),
        'norm_H': torch.norm(observables['H']).item(),
        'norm_q': torch.norm(observables['q']).item(),
    }


if __name__ == "__main__":
    # Test computation
    d, n, r = 32, 128, 5
    device = torch.device('cpu')
    
    # Create test tensors
    W = torch.randn(d, n, device=device)
    A = torch.randn(n, d, device=device)
    b = torch.randn(n, device=device)
    C = torch.randn(d, device=device)
    U = torch.randn(n, r, device=device)
    U, _ = torch.linalg.qr(U)  # Make orthonormal
    v = torch.randn(n, device=device)
    v = v / torch.norm(v)  # Make unit norm
    Lambda = torch.diag(torch.linspace(1, 0.1, r, device=device))
    
    eta = 0.1
    sigma2_y = 0.1
    
    # Compute observables
    obs = compute_branch_b_observables(W, A, b, C, U, v, Lambda, eta, sigma2_y)
    print("Computed 13 observables successfully")
    print(f"M shape: {obs['M'].shape}, S shape: {obs['S'].shape}")
    
    # Compute time derivatives
    derivs = compute_branch_b_time_derivatives(obs, 0.01, 0.01, 0.01, 0.01)
    print("Computed 13 time derivatives successfully")
    
    # Compute convergence
    conv = compute_convergence_metrics(derivs)
    print(f"Max gradient norm: {conv['max_gradient_norm']:.6e}")
    
    # Pack for history
    hist = pack_observables_for_history(obs)
    print(f"Packed {len(hist)} scalar observables")
