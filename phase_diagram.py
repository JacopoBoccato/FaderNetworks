#!/usr/bin/env python3
"""
Compare Branch-B theory against a trained strict linear Fader.

This script:
1. Builds the same teacher model / parameters used in 3d_sim.py.
2. Initializes a strict linear Fader with repository modules.
3. Computes the full Branch-B observable state at initialization.
4. Integrates the theoretical ODE from that same initial observable state.
5. Trains the microscopic linear Fader with minibatch saddle updates under
   that same game.
6. Plots theoretical trajectories vs measured trajectories for all observables.
"""

import argparse
from dataclasses import dataclass
import os
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from src.model import AutoEncoder, LatentDiscriminator, sequence_reconstruction_loss
from src.branch_b_observables import compute_branch_b_observables


D_DIM = 3
R_DIM = 3
N_DIM = 32

assert D_DIM == R_DIM, "This comparison script assumes D_DIM == R_DIM."

H_DIRECTION = np.array([0.5, 0.5, 0.0], dtype=float)
H_DIRECTION = H_DIRECTION / max(np.linalg.norm(H_DIRECTION), 1e-12)
ETA_FRACTION = 0.5

@dataclass(frozen=True)
class ExperimentConfig:
    # Shared theory / Fader controls.
    noise_total: float = 0.5
    lambda_reg: float = 0.5
    lam_sig: float = 15.0
    lambda_C: float = 0.1
    alpha_C: float = 1.0
    eta_clf: float = 100.0
    gamma0: float = 0.0
    gamma_mu: float = 0.0
    h_scale: float = 1.0

    # Training / measurement controls.
    n_epochs: int = 2000
    n_samples: int = 100000
    batch_size: int = 128
    epoch_size: int = 0
    learning_rate: float = 5e-4
    measure_every_batches: int = 1
    lambda_schedule: int = 0

    # Theory integration controls.
    theory_points: int = 4000

    # Reproducibility / runtime controls.
    seed: int = 0
    teacher_seed: int = 0
    device: str = "cpu"


# Edit this block. It is the single source of truth for both the Fader
# training game and the ODE parameters.
EXPERIMENT_CONFIG = ExperimentConfig()

OUT_PLOT = "branch_b_theory_vs_training.png"
OUT_LOSS = "branch_b_training_loss.png"
OUT_DATA = "branch_b_theory_vs_training.npz"

MAT_DD = D_DIM * D_DIM
MAT_RD = R_DIM * D_DIM
VEC_D = D_DIM
VEC_R = R_DIM


def build_shared_params(config):
    eta = max(ETA_FRACTION * config.noise_total, 1e-8)
    g = max((1.0 - ETA_FRACTION) * config.noise_total, 1e-8)
    return {
        "noise_total": config.noise_total,
        "learning_rate": config.learning_rate,
        "lambda_reg": config.lambda_reg,
        "lam_sig": config.lam_sig,
        "lambda_C": config.lambda_C,
        "alpha_C": max(float(config.alpha_C), 0.0),
        "eta_clf": max(config.eta_clf, 1e-8),
        "gamma0": float(config.gamma0),
        "gamma_mu": max(float(config.gamma_mu), 0.0),
        "h_scale": config.h_scale,
        "h_vec": config.h_scale * H_DIRECTION,
        "eta": eta,
        "g": g,
    }


def build_teacher(n, r, h_scale, seed=0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, r + 1)))
    U = q[:, :r]
    v_perp = q[:, r]

    h = h_scale * H_DIRECTION
    h_norm2 = float(np.dot(h, h))
    if h_norm2 >= 1.0:
        h = h / max(np.linalg.norm(h), 1e-12) * 0.999
        h_norm2 = float(np.dot(h, h))

    v = U @ h + np.sqrt(max(1.0 - h_norm2, 1e-12)) * v_perp
    v = v / max(np.linalg.norm(v), 1e-12)
    return U, v


def generate_dataset(config, shared_params, n, r):
    eta = shared_params["eta"]
    g = shared_params["g"]

    U, v = build_teacher(n, r, config.h_scale, seed=config.teacher_seed)
    rng = np.random.default_rng(config.teacher_seed)

    c = rng.standard_normal((config.n_samples, r))
    y = np.sqrt(g) * rng.standard_normal((config.n_samples, 1))
    a = rng.standard_normal((config.n_samples, n))

    c_scaled = np.sqrt(max(shared_params["lam_sig"], 1e-12)) * c
    X = (U @ c_scaled.T).T + (v[:, None] @ y.T).T + np.sqrt(eta) * a

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.squeeze(), dtype=torch.float32).unsqueeze(-1)
    lam = shared_params["lam_sig"] * np.ones(r, dtype=float)
    return X, y, U, v, lam


def make_linear_fader(device):
    params = SimpleNamespace(
        seq_len=N_DIM,
        x_type="continuous",
        n_attr=1,
        label_type="continuous",
        encoder_hidden_dims=[D_DIM],
        decoder_hidden_dims=[],
        dis_hidden_dims=[],
        hid_dim=D_DIM,
        n_amino=0,
    )

    ae = AutoEncoder(params).to(device)
    lat_dis = LatentDiscriminator(params).to(device)

    enc_linear = [m for m in ae.encoder.modules() if isinstance(m, nn.Linear)]
    dec_linear = [m for m in ae.decoder.modules() if isinstance(m, nn.Linear)]
    clf_linear = [m for m in lat_dis.net.modules() if isinstance(m, nn.Linear)]

    assert len(enc_linear) == 1
    assert len(dec_linear) == 1
    assert len(clf_linear) == 1

    enc_linear[0].bias.data.zero_()
    dec_linear[0].bias.data.zero_()
    clf_linear[0].bias.data.zero_()
    enc_linear[0].bias.requires_grad_(False)
    dec_linear[0].bias.requires_grad_(False)
    clf_linear[0].bias.requires_grad_(False)
    return ae, lat_dis


def extract_linear_matrices(ae, lat_dis, device):
    enc = [m for m in ae.encoder.modules() if isinstance(m, nn.Linear)][0]
    dec = [m for m in ae.decoder.modules() if isinstance(m, nn.Linear)][0]
    clf = [m for m in lat_dis.net.modules() if isinstance(m, nn.Linear)][0]

    W = enc.weight
    dec_w = dec.weight
    A = dec_w[:, :D_DIM]
    b = dec_w[:, D_DIM]
    C = clf.weight.view(-1)
    return W.to(device), A.to(device), b.to(device), C.to(device)


def pack_state(M, s, N, a, beta, rho, C, Q, T, u, t, B, m):
    return np.concatenate(
        [
            M.reshape(-1),
            s.reshape(-1),
            N.reshape(-1),
            a.reshape(-1),
            beta.reshape(-1),
            np.array([rho]),
            C.reshape(-1),
            Q.reshape(-1),
            T.reshape(-1),
            u.reshape(-1),
            t.reshape(-1),
            B.reshape(-1),
            np.array([m]),
        ]
    )


def unpack_state(X):
    idx = 0

    def take(size):
        nonlocal idx
        out = X[idx:idx + size]
        idx += size
        return out

    M = take(D_DIM * R_DIM).reshape(D_DIM, R_DIM)
    s = take(VEC_D)
    N = take(R_DIM * D_DIM).reshape(R_DIM, D_DIM)
    a = take(VEC_D)
    beta = take(VEC_R)
    rho = take(1)[0]
    C = take(VEC_D)
    Q = take(MAT_DD).reshape(D_DIM, D_DIM)
    T = take(MAT_DD).reshape(D_DIM, D_DIM)
    u = take(VEC_D)
    t = take(VEC_D)
    B = take(MAT_DD).reshape(D_DIM, D_DIM)
    m = take(1)[0]
    return M, s, N, a, beta, rho, C, Q, T, u, t, B, m


def observables_to_state(observables):
    return pack_state(
        observables["M"].detach().cpu().numpy(),
        observables["s"].detach().cpu().numpy(),
        observables["N"].detach().cpu().numpy(),
        observables["a"].detach().cpu().numpy(),
        observables["beta"].detach().cpu().numpy(),
        float(observables["rho"].detach().cpu().item()),
        observables["C"].detach().cpu().numpy(),
        observables["Q"].detach().cpu().numpy(),
        observables["T"].detach().cpu().numpy(),
        observables["u"].detach().cpu().numpy(),
        observables["t"].detach().cpu().numpy(),
        observables["B"].detach().cpu().numpy(),
        float(observables["m"].detach().cpu().item()),
    )


def sym(A):
    return 0.5 * (A + A.T)


def q_scale(Q):
    return np.sqrt(max(np.trace(sym(Q)), 1e-8))


def reconstruction_error(state, params):
    M, s, N, a, beta, rho, C, Q, T, u, t, B, m = unpack_state(state)
    lam_sig = params["lam_sig"]
    g = params["g"]
    eta = params["eta"]

    Lambda = lam_sig * np.eye(R_DIM)
    S = M @ Lambda @ M.T + g * np.outer(s, s) + eta * Q
    G = N.T @ Lambda @ M.T + g * np.outer(a, s) + eta * B
    trace_sigma = np.trace(Lambda) + g + eta * D_DIM
    err = trace_sigma + np.trace(T @ S) - 2.0 * np.trace(G) + 2.0 * g * (u @ s - rho) + g * m
    return float(np.real(err))


def scalarize_state(state, params):
    M, s, N, a, beta, rho, C, Q, T, u, t, B, m = unpack_state(state)
    scale = q_scale(Q)
    return {
        "norm_M": float(np.linalg.norm(M, ord="fro")),
        "norm_s": float(np.linalg.norm(s)),
        "norm_N": float(np.linalg.norm(N, ord="fro")),
        "norm_a": float(np.linalg.norm(a)),
        "norm_beta": float(np.linalg.norm(beta)),
        "rho": float(rho),
        "norm_C": float(np.linalg.norm(C)),
        "norm_Q": float(np.linalg.norm(Q, ord="fro")),
        "norm_T": float(np.linalg.norm(T, ord="fro")),
        "norm_u": float(np.linalg.norm(u)),
        "norm_t": float(np.linalg.norm(t)),
        "norm_B": float(np.linalg.norm(B, ord="fro")),
        "m": float(m),
        "reconstruction_error": reconstruction_error(state, params),
        "M_tilde": float(np.linalg.norm(M, ord="fro") / scale),
        "N_tilde": float(np.linalg.norm(N, ord="fro") / scale),
    }


def scalarize_observables(observables, rec_loss):
    Q = observables["Q"].detach().cpu().numpy()
    scale = q_scale(Q)
    return {
        "norm_M": float(torch.norm(observables["M"], p="fro").item()),
        "norm_s": float(torch.norm(observables["s"]).item()),
        "norm_N": float(torch.norm(observables["N"], p="fro").item()),
        "norm_a": float(torch.norm(observables["a"]).item()),
        "norm_beta": float(torch.norm(observables["beta"]).item()),
        "rho": float(observables["rho"].item()),
        "norm_C": float(torch.norm(observables["C"]).item()),
        "norm_Q": float(torch.norm(observables["Q"], p="fro").item()),
        "norm_T": float(torch.norm(observables["T"], p="fro").item()),
        "norm_u": float(torch.norm(observables["u"]).item()),
        "norm_t": float(torch.norm(observables["t"]).item()),
        "norm_B": float(torch.norm(observables["B"], p="fro").item()),
        "m": float(observables["m"].item()),
        "reconstruction_error": float(rec_loss),
        "M_tilde": float(torch.norm(observables["M"], p="fro").item() / scale),
        "N_tilde": float(torch.norm(observables["N"], p="fro").item() / scale),
    }


def classifier_strength(tau, params):
    """Classifier coupling Gamma(t) from the paper's saturating schedule."""
    gamma_inf = float(params["eta_clf"])
    gamma0 = float(params.get("gamma0", gamma_inf))
    gamma_mu = float(params.get("gamma_mu", 0.0))
    if gamma_mu <= 0.0:
        return gamma_inf
    tau = max(float(tau), 0.0)
    return gamma_inf - (gamma_inf - gamma0) * np.exp(-gamma_mu * tau)


def rhs(tau, X, params):
    M, s, N, a, beta, rho, C, Q, T, u, t, B, m = unpack_state(X)

    lam_sig = params["lam_sig"]
    g = params["g"]
    eta = params["eta"]
    lW = params["lambda_reg"]
    lA = params["lambda_reg"]
    lb = params["lambda_reg"]
    lC = params["lambda_C"]
    alpha_C = params["alpha_C"]
    gamma_t = classifier_strength(tau, params)
    h = params["h_vec"]

    Lambda = lam_sig * np.eye(R_DIM)
    D = (lam_sig + eta) * np.eye(R_DIM)
    kappa = g + eta
    CCt = np.outer(C, C)

    S = M @ Lambda @ M.T + g * np.outer(s, s) + eta * Q
    G = N.T @ Lambda @ M.T + g * np.outer(a, s) + eta * B
    J = N.T @ Lambda @ N + g * np.outer(a, a) + eta * T
    H = M @ Lambda @ beta + g * rho * s + eta * t
    q = N.T @ Lambda @ beta + g * rho * a + eta * u

    aux = T @ S - G + g * np.outer(u, s)

    dM = (
        -2 * (T @ M @ D - N.T @ D + g * np.outer(T @ s - a + u, h))
        + 2 * gamma_t * (CCt @ (M @ D + g * np.outer(s, h)))
        - 2 * gamma_t * g * np.outer(C, h)
        - 2 * lW * M
    )
    ds = (
        -2 * ((T @ M - N.T) @ (Lambda @ h) + kappa * (T @ s) - kappa * a + g * u)
        + 2 * gamma_t * (CCt @ (M @ (Lambda @ h) + kappa * s))
        - 2 * gamma_t * g * C
        - 2 * lW * s
    )
    dN = -2 * (N @ S - D @ M.T + g * np.outer(beta - h, s)) - 2 * lA * N
    da = -2 * (S @ a - M @ (Lambda @ h) - kappa * s + g * rho * s) - 2 * lA * a
    dbeta = -2 * g * (N @ s - h + beta) - 2 * lb * beta
    drho = -2 * g * (a @ s - 1.0 + rho) - 2 * lb * rho
    dC = -2 * alpha_C * (gamma_t * (S @ C - g * s) + lC * C)
    dQ = (
        -2 * aux
        - 2 * aux.T
        + 2 * gamma_t * (CCt @ S + S @ CCt)
        - 2 * gamma_t * g * (np.outer(C, s) + np.outer(s, C))
        - 4 * lW * Q
    )
    dT = -2 * aux - 2 * aux.T - 4 * lA * T
    du = -2 * (S @ u - H + g * m * s) - 2 * g * (T @ s - a + u) - 2 * (lA + lb) * u
    dt_ = (
        -2 * (T @ H - q + g * rho * u)
        + 2 * gamma_t * (CCt @ H)
        - 2 * gamma_t * g * rho * C
        - 2 * g * (B.T @ s - s + t)
        - 2 * (lW + lb) * t
    )
    dB = (
        -2 * (S @ B - S + g * np.outer(s, t))
        - 2 * (G @ T - J + g * np.outer(a, u))
        + 2 * gamma_t * (G @ CCt)
        - 2 * gamma_t * g * np.outer(a, C)
        - 2 * (lA + lW) * B
    )
    dm = -4 * g * (u @ s - rho + m) - 4 * lb * m
    return pack_state(dM, ds, dN, da, dbeta, drho, dC, dQ, dT, du, dt_, dB, dm)


def integrate_theory(x0, params, t_eval):
    sol = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        x0,
        t_eval=t_eval,
        args=(params,),
        rtol=1e-6,
        atol=1e-8,
        method="RK45",
    )
    if not sol.success:
        raise RuntimeError(f"Theory integration failed: {sol.message}")
    return sol.y.T


def compute_measured_observables(ae, lat_dis, X, y, U, v, lam, eta, g, device):
    ae.eval()
    lat_dis.eval()
    with torch.no_grad():
        W, A, b, C = extract_linear_matrices(ae, lat_dis, device)
        U_t = torch.tensor(U, dtype=torch.float32, device=device)
        v_t = torch.tensor(v, dtype=torch.float32, device=device)
        Lambda_t = torch.tensor(np.diag(lam), dtype=torch.float32, device=device)
        observables = compute_branch_b_observables(W, A, b, C, U_t, v_t, Lambda_t, eta, g)
        X_t = X.to(device)
        y_t = y.to(device)
        rec_loss = torch.mean((ae(X_t, y_t)[1][-1] - X_t) ** 2).item()
    return observables, rec_loss


def microscopic_objectives(ae, lat_dis, X_dev, y_dev, gamma_t, params, device):
    enc_outputs, dec_outputs = ae(X_dev, y_dev)
    recon = dec_outputs[-1]
    preds = lat_dis(enc_outputs[-1])
    rec_loss = sequence_reconstruction_loss(recon, X_dev, x_type="continuous")
    clf_loss = torch.mean((preds - y_dev) ** 2)

    W, A, b, C = extract_linear_matrices(ae, lat_dis, device)
    reg_ae = params["lambda_reg"] * ((W ** 2).sum() + (A ** 2).sum() + (b ** 2).sum())
    reg_c = params["lambda_C"] * (C ** 2).sum()

    ae_objective = rec_loss - gamma_t * clf_loss + reg_ae
    c_objective = gamma_t * clf_loss + reg_c
    return ae_objective, c_objective, rec_loss, clf_loss


def microscopic_saddle_step(ae, lat_dis, batch_x, batch_y, tau, params, device):
    ae.train()
    lat_dis.train()

    gamma_t = classifier_strength(tau, params)
    ae_objective, c_objective, rec_loss, clf_loss = microscopic_objectives(
        ae, lat_dis, batch_x, batch_y, gamma_t, params, device
    )

    ae_params = [p for p in ae.parameters() if p.requires_grad]
    c_params = [p for p in lat_dis.parameters() if p.requires_grad]

    ae_grads = torch.autograd.grad(ae_objective, ae_params, retain_graph=True)
    c_grads = torch.autograd.grad(c_objective, c_params)

    with torch.no_grad():
        for param, grad in zip(ae_params, ae_grads):
            param -= float(params["learning_rate"]) * grad
        for param, grad in zip(c_params, c_grads):
            param -= float(params["learning_rate"]) * float(params["alpha_C"]) * grad

    return (
        float(ae_objective.item()),
        float(c_objective.item()),
        float(rec_loss.item()),
        float(clf_loss.item()),
        float(gamma_t),
    )


def train_and_measure(X, y, U, v, lam, params, config, device):
    ae, lat_dis = make_linear_fader(device)

    X_dev = X.to(device)
    y_dev = y.to(device)
    n_samples = X_dev.size(0)
    batch_size = min(max(int(config.batch_size), 1), n_samples)
    epoch_size = int(config.epoch_size) if int(config.epoch_size) > 0 else int(n_samples)
    n_batches_per_epoch = max(int(np.ceil(epoch_size / batch_size)), 1)
    total_steps = max(int(config.n_epochs) * n_batches_per_epoch, 1)
    measure_every_steps = max(int(config.measure_every_batches), 1)
    joint_update_dt = float(config.learning_rate)

    measured = []
    times = [0.0]
    sgd_times = [0.0]
    epoch_ae_losses = []
    epoch_clf_losses = []
    epoch_rec_losses = []
    gamma_history = []

    initial_obs, initial_rec = compute_measured_observables(
        ae, lat_dis, X, y, U, v, lam, params["eta"], params["g"], device
    )
    measured.append(scalarize_observables(initial_obs, initial_rec))

    elapsed_time = 0.0
    step_idx = 0
    for _ in range(int(config.n_epochs)):
        order = torch.randperm(n_samples, device=device)
        offset = 0

        for _ in range(n_batches_per_epoch):
            if offset + batch_size > n_samples:
                order = torch.randperm(n_samples, device=device)
                offset = 0

            batch_idx = order[offset:offset + batch_size]
            offset += batch_size

            batch_x = X_dev[batch_idx]
            batch_y = y_dev[batch_idx]

            ae_obj, c_obj, rec_obj, clf_obj, gamma_t = microscopic_saddle_step(
                ae, lat_dis, batch_x, batch_y, elapsed_time, params, device
            )

            step_idx += 1
            elapsed_time += joint_update_dt

            if step_idx % measure_every_steps == 0 or step_idx == total_steps:
                obs, rec = compute_measured_observables(
                    ae,
                    lat_dis,
                    X,
                    y,
                    U,
                    v,
                    lam,
                    params["eta"],
                    params["g"],
                    device,
                )
                measured.append(scalarize_observables(obs, rec))
                times.append(elapsed_time)
                sgd_times.append(elapsed_time)

        gamma_epoch = classifier_strength(elapsed_time, params)
        ae_epoch, c_epoch, rec_epoch, _ = microscopic_objectives(
            ae, lat_dis, X_dev, y_dev, gamma_epoch, params, device
        )
        epoch_ae_losses.append(float(ae_epoch.item()))
        epoch_clf_losses.append(float(c_epoch.item()))
        epoch_rec_losses.append(float(rec_epoch.item()))
        gamma_history.append(float(gamma_epoch))

    return (
        ae,
        lat_dis,
        np.asarray(times, dtype=float),
        np.asarray(sgd_times, dtype=float),
        {
            "ae_loss": np.asarray(epoch_ae_losses, dtype=float),
            "clf_loss": np.asarray(epoch_clf_losses, dtype=float),
            "rec_loss": np.asarray(epoch_rec_losses, dtype=float),
            "gamma": np.asarray(gamma_history, dtype=float),
        },
        measured,
        observables_to_state(initial_obs),
        float(joint_update_dt),
    )


def plot_comparison(times, theory_times_dense, theory_hist_dense, measured_hist, params, out_file):
    metrics = [
        ("norm_M", "||M||_F"),
        ("norm_s", "||s||"),
        ("norm_N", "||N||_F"),
        ("norm_a", "||a||"),
        ("norm_beta", "||beta||"),
        ("rho", "rho"),
        ("norm_C", "||C||"),
        ("norm_Q", "||Q||_F"),
        ("norm_T", "||T||_F"),
        ("norm_u", "||u||"),
        ("norm_t", "||t||"),
        ("norm_B", "||B||_F"),
        ("m", "m = ||b||^2"),
        ("reconstruction_error", "reconstruction error"),
        ("M_tilde", "||M|| / sqrt(tr(Q))"),
        ("N_tilde", "||N|| / sqrt(tr(Q))"),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    axes = axes.ravel()

    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx]
        ax.plot(theory_times_dense, theory_hist_dense[key], label="theory", linewidth=2.0)
        ax.plot(times, measured_hist[key], label="measured", linewidth=1.5, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel(r"$\tau$ (lr-scaled joint updates)")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend()

    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        "Branch-B Theory vs Measured Linear Fader Dynamics\n"
        f"noise_total={params['noise_total']}, lambda_reg={params['lambda_reg']}, "
        f"lam_sig={params['lam_sig']}, lambda_C={params['lambda_C']}, "
        f"dt={params['learning_rate']}, alpha_C={params['alpha_C']}, "
        f"Gamma_inf={params['eta_clf']}, Gamma0={params['gamma0']}, "
        f"gamma_mu={params['gamma_mu']}, h_scale={params['h_scale']}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_file, dpi=160, bbox_inches="tight")


def plot_loss_curve(loss_history, params, out_file=OUT_LOSS):
    """Plot the repository-style training objective and reconstruction proxy."""
    ae_losses = np.asarray(loss_history["ae_loss"], dtype=float)
    clf_losses = np.asarray(loss_history["clf_loss"], dtype=float)
    rec_losses = np.asarray(loss_history["rec_loss"], dtype=float)
    gamma_values = np.asarray(loss_history["gamma"], dtype=float)
    if len(ae_losses) == 0:
        raise ValueError("No epoch losses available to plot.")

    epochs = np.arange(1, len(ae_losses) + 1, dtype=int)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), sharex=True)
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes:
        ax.set_facecolor("#0f0f23")
        ax.grid(True, alpha=0.25)

    axes[0].plot(epochs, ae_losses, color="#f1c40f", lw=2.0, label="AE descent objective")
    axes[0].plot(epochs, clf_losses, color="#3498db", lw=1.8, label="classifier descent objective")
    axes[0].set_ylabel("train objective")
    axes[0].set_title("Microscopic saddle objectives")
    axes[0].legend()

    axes[1].plot(epochs, rec_losses, color="#2ecc71", lw=2.0, label="full-dataset reconstruction MSE")
    axes[1].set_ylabel("reconstruction MSE")
    axes[1].set_title("Convergence proxy")
    axes[1].legend()

    if len(rec_losses) >= 4:
        win = min(10, len(rec_losses) // 2)
        if win >= 2:
            prev = rec_losses[-2 * win:-win]
            last = rec_losses[-win:]
            prev_mean = float(np.mean(prev))
            last_mean = float(np.mean(last))
            rel_improve = abs(prev_mean - last_mean) / max(abs(prev_mean), 1e-12)
            stability = float(np.std(last) / max(abs(last_mean), 1e-12))
            axes[1].axvspan(max(len(rec_losses) - 2 * win, 1), len(rec_losses), color="#9b59b6", alpha=0.12, label="convergence window")
            axes[1].text(
                0.98,
                0.02,
                f"final rec={rec_losses[-1]:.4g}\nrel_improve={rel_improve:.3g}\nstability={stability:.3g}",
                transform=axes[1].transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", fc="#0f0f23", ec="#666", alpha=0.92),
            )

    axes[2].plot(epochs, gamma_values, color="#e67e22", lw=2.0, label=r"$\Gamma(\tau)$")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("classifier strength")
    axes[2].set_title("Time-dependent classifier schedule")
    axes[2].legend()

    fig.suptitle(
        "Linear Fader saddle-game diagnostics\n"
        f"lambda_reg={params['lambda_reg']}, lambda_C={params['lambda_C']}, "
        f"dt={params['learning_rate']}, alpha_C={params['alpha_C']}, "
        f"Gamma_inf={params['eta_clf']}, Gamma0={params['gamma0']}, gamma_mu={params['gamma_mu']}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_file, dpi=160, bbox_inches="tight")


def parse_config():
    base = EXPERIMENT_CONFIG
    parser = argparse.ArgumentParser(description="Compare Branch-B ODE against trained linear Fader observables.")
    parser.add_argument("--device", type=str, default=base.device, choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=base.seed)
    parser.add_argument("--teacher_seed", type=int, default=base.teacher_seed)
    parser.add_argument("--n_epochs", type=int, default=base.n_epochs)
    parser.add_argument("--n_samples", type=int, default=base.n_samples)
    parser.add_argument("--batch_size", type=int, default=base.batch_size)
    parser.add_argument("--epoch_size", type=int, default=base.epoch_size)
    parser.add_argument("--learning_rate", type=float, default=base.learning_rate)
    parser.add_argument("--measure_every_batches", type=int, default=base.measure_every_batches)
    parser.add_argument("--lambda_schedule", type=int, default=base.lambda_schedule)
    parser.add_argument("--theory_points", type=int, default=base.theory_points)
    parser.add_argument("--noise_total", type=float, default=base.noise_total)
    parser.add_argument("--lambda_reg", type=float, default=base.lambda_reg)
    parser.add_argument("--lam_sig", type=float, default=base.lam_sig)
    parser.add_argument("--lambda_C", type=float, default=base.lambda_C)
    parser.add_argument("--alpha_C", type=float, default=base.alpha_C)
    parser.add_argument("--eta_clf", type=float, default=base.eta_clf)
    parser.add_argument("--gamma0", type=float, default=base.gamma0)
    parser.add_argument("--gamma_mu", type=float, default=base.gamma_mu)
    parser.add_argument("--h_scale", type=float, default=base.h_scale)
    parser.add_argument("--out_plot", type=str, default=OUT_PLOT)
    parser.add_argument("--out_loss", type=str, default=OUT_LOSS)
    parser.add_argument("--out_data", type=str, default=OUT_DATA)
    args = parser.parse_args()
    config = ExperimentConfig(
        noise_total=args.noise_total,
        lambda_reg=args.lambda_reg,
        lam_sig=args.lam_sig,
        lambda_C=args.lambda_C,
        alpha_C=args.alpha_C,
        eta_clf=args.eta_clf,
        gamma0=args.gamma0,
        gamma_mu=args.gamma_mu,
        h_scale=args.h_scale,
        n_epochs=args.n_epochs,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        epoch_size=args.epoch_size,
        learning_rate=args.learning_rate,
        measure_every_batches=args.measure_every_batches,
        lambda_schedule=args.lambda_schedule,
        theory_points=args.theory_points,
        seed=args.seed,
        teacher_seed=args.teacher_seed,
        device=args.device,
    )
    return config, args.out_plot, args.out_loss, args.out_data


def main():
    config, out_plot, out_loss, out_data = parse_config()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    params = build_shared_params(config)
    X, y, U, v, lam = generate_dataset(config, params, N_DIM, R_DIM)

    _, _, times, sgd_times, loss_history, measured_list, x0, matched_dt = train_and_measure(
        X,
        y,
        U,
        v,
        lam,
        params,
        config,
        device=device,
    )

    theory_t_final = float(times[-1])
    theory_times_dense = np.linspace(0.0, theory_t_final, int(config.theory_points))
    theory_states_dense = integrate_theory(x0, params, theory_times_dense)
    theory_list_dense = [scalarize_state(x, params) for x in theory_states_dense]

    keys = list(measured_list[0].keys())
    measured_hist = {k: np.array([row[k] for row in measured_list], dtype=float) for k in keys}
    theory_hist_dense = {k: np.array([row[k] for row in theory_list_dense], dtype=float) for k in keys}
    theory_hist = {k: np.interp(times, theory_times_dense, theory_hist_dense[k]) for k in keys}

    plot_comparison(times, theory_times_dense, theory_hist_dense, measured_hist, params, out_plot)
    plot_loss_curve(loss_history, params, out_file=out_loss)

    np.savez(
        out_data,
        times=times,
        times_sgd=sgd_times,
        epoch_ae_losses=loss_history["ae_loss"],
        epoch_clf_losses=loss_history["clf_loss"],
        epoch_rec_losses=loss_history["rec_loss"],
        gamma_history=loss_history["gamma"],
        theory_times_dense=theory_times_dense,
        **{f"measured_{k}": v for k, v in measured_hist.items()},
        **{f"theory_{k}": v for k, v in theory_hist.items()},
        **{f"theory_dense_{k}": v for k, v in theory_hist_dense.items()},
        matched_dt=matched_dt,
        alpha_AE=params["learning_rate"],
        alpha_C=params["alpha_C"],
        gamma0=params["gamma0"],
        gamma_inf=params["eta_clf"],
        gamma_mu=params["gamma_mu"],
        measure_every_steps=config.measure_every_batches,
        batch_size=config.batch_size,
        epoch_size=config.epoch_size,
        n_steps=config.n_epochs,
        noise_total=params["noise_total"],
        lambda_reg=params["lambda_reg"],
        lam_sig=params["lam_sig"],
        lambda_C=params["lambda_C"],
        eta_clf=params["eta_clf"],
        h_scale=params["h_scale"],
        eta=params["eta"],
        g=params["g"],
    )
    print(f"Shared config -> {config}")
    print(f"Saved -> {out_plot}")
    print(f"Saved -> {out_loss}")
    print(f"Saved -> {out_data}")


if __name__ == "__main__":
    main()
