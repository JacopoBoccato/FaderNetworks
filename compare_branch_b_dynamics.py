#!/usr/bin/env python3
"""
Compare Branch-B theory against a trained strict linear Fader.

This script:
1. Builds the same teacher model / parameters used in 3d_sim.py.
2. Initializes a strict linear Fader with repository modules.
3. Computes the full Branch-B observable state at initialization.
4. Integrates the theoretical ODE from that same initial observable state.
5. Trains the microscopic linear Fader under the same fixed game.
6. Plots theoretical trajectories vs measured trajectories for all observables.
"""

import argparse
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from src.model import AutoEncoder, LatentDiscriminator
from src.branch_b_observables import compute_branch_b_observables


D_DIM = 3
R_DIM = 3
N_DIM = 32

assert D_DIM == R_DIM, "This comparison script assumes D_DIM == R_DIM."

H_DIRECTION = np.array([0.5, 0.5, 0.0], dtype=float)
H_DIRECTION = H_DIRECTION / max(np.linalg.norm(H_DIRECTION), 1e-12)
ETA_FRACTION = 0.5

FIXED_VALUES = {
    "noise_total": 0.5,
    "lambda_reg": 0.1,
    "lam_sig": 5.0,
    "lambda_C": 0.01,
    "alpha_C": 1.0,
    "eta_clf": 1.0,
    "h_scale": 1.0,
}

TRAIN_CONFIG = {
    "n_epochs": 50000,
    "n_samples": 50000,
    "batch_size": 16384,
    "learning_rate": 5e-6,
}

OUT_PLOT = "branch_b_theory_vs_training.png"
OUT_DATA = "branch_b_theory_vs_training.npz"
THEORY_T_FINAL = 500.0
THEORY_POINTS = 20000

MAT_DD = D_DIM * D_DIM
MAT_RD = R_DIM * D_DIM
VEC_D = D_DIM
VEC_R = R_DIM


def build_params(noise_total, lambda_reg, lam_sig, lambda_C, alpha_C, eta_clf, h_scale):
    eta = max(ETA_FRACTION * noise_total, 1e-8)
    g = max((1.0 - ETA_FRACTION) * noise_total, 1e-8)
    return {
        "noise_total": noise_total,
        "lambda_reg": lambda_reg,
        "lam_sig": lam_sig,
        "lambda_C": lambda_C,
        "alpha_C": max(alpha_C, 1e-8),
        "eta_clf": max(eta_clf, 1e-8),
        "h_scale": h_scale,
        "h_vec": h_scale * H_DIRECTION,
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


def generate_dataset(n_samples, n, r, lam_sig, noise_total, h_scale, seed=0):
    params = build_params(noise_total, 0.0, lam_sig, 0.0, 1.0, 1.0, h_scale)
    eta = params["eta"]
    g = params["g"]

    U, v = build_teacher(n, r, h_scale, seed=seed)
    rng = np.random.default_rng(seed)

    c = rng.standard_normal((n_samples, r))
    y = np.sqrt(g) * rng.standard_normal((n_samples, 1))
    a = rng.standard_normal((n_samples, n))

    c_scaled = np.sqrt(max(lam_sig, 1e-12)) * c
    X = (U @ c_scaled.T).T + (v[:, None] @ y.T).T + np.sqrt(eta) * a

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.squeeze(), dtype=torch.float32).unsqueeze(-1)
    lam = lam_sig * np.ones(r, dtype=float)
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
    eta_clf = params["eta_clf"]
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
        + 2 * eta_clf * (CCt @ (M @ D + g * np.outer(s, h)))
        - 2 * eta_clf * g * np.outer(C, h)
        - 2 * lW * M
    )
    ds = (
        -2 * ((T @ M - N.T) @ (Lambda @ h) + kappa * (T @ s) - kappa * a + g * u)
        + 2 * eta_clf * (CCt @ (M @ (Lambda @ h) + kappa * s))
        - 2 * eta_clf * g * C
        - 2 * lW * s
    )
    dN = -2 * (N @ S - D @ M.T + g * np.outer(beta - h, s)) - 2 * lA * N
    da = -2 * (S @ a - M @ (Lambda @ h) - kappa * s + g * rho * s) - 2 * lA * a
    dbeta = -2 * g * (N @ s - h + beta) - 2 * lb * beta
    drho = -2 * g * (a @ s - 1.0 + rho) - 2 * lb * rho
    dC = -2 * alpha_C * (eta_clf * (S @ C - g * s) + lC * C)
    dQ = (
        -2 * aux
        - 2 * aux.T
        + 2 * eta_clf * (CCt @ S + S @ CCt)
        - 2 * eta_clf * g * (np.outer(C, s) + np.outer(s, C))
        - 4 * lW * Q
    )
    dT = -2 * aux - 2 * aux.T - 4 * lA * T
    du = -2 * (S @ u - H + g * m * s) - 2 * g * (T @ s - a + u) - 2 * (lA + lb) * u
    dt_ = (
        -2 * (T @ H - q + g * rho * u)
        + 2 * eta_clf * (CCt @ H)
        - 2 * eta_clf * g * rho * C
        - 2 * g * (B.T @ s - s + t)
        - 2 * (lW + lb) * t
    )
    dB = (
        -2 * (S @ B - S + g * np.outer(s, t))
        - 2 * (G @ T - J + g * np.outer(a, u))
        + 2 * eta_clf * (G @ CCt)
        - 2 * eta_clf * g * np.outer(a, C)
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


def train_and_measure(X, y, U, v, lam, params, n_epochs, lr, batch_size, device):
    ae, lat_dis = make_linear_fader(device)
    ae_opt = torch.optim.SGD([p for p in ae.parameters() if p.requires_grad], lr=lr)
    c_opt = torch.optim.SGD([p for p in lat_dis.parameters() if p.requires_grad], lr=lr * params["alpha_C"])

    X_dev = X.to(device)
    y_dev = y.to(device)
    n_samples = X_dev.size(0)
    batch_size = min(int(batch_size), n_samples)

    measured = []
    times = [0.0]

    initial_obs, initial_rec = compute_measured_observables(
        ae, lat_dis, X, y, U, v, lam, params["eta"], params["g"], device
    )
    measured.append(scalarize_observables(initial_obs, initial_rec))

    elapsed_time = 0.0
    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(n_samples, device=X_dev.device)
        n_batches = 0
        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            batch_x = X_dev[idx]
            batch_y = y_dev[idx]

            # Full classifier pressure from the first update; no schedule or warmup.
            c_opt.zero_grad()
            with torch.no_grad():
                z_det = ae.encode(batch_x)[-1]
            y_hat = lat_dis(z_det)
            _, _, _, C = extract_linear_matrices(ae, lat_dis, device)
            loss_c = params["eta_clf"] * torch.mean((y_hat - batch_y) ** 2) + params["lambda_C"] * (torch.norm(C) ** 2)
            loss_c.backward()
            c_opt.step()

            ae_opt.zero_grad()
            enc_outputs, dec_outputs = ae(batch_x, batch_y)
            z = enc_outputs[-1]
            x_hat = dec_outputs[-1]
            y_hat_adv = lat_dis(z)
            W, A, b, _ = extract_linear_matrices(ae, lat_dis, device)
            rec_loss = torch.mean((x_hat - batch_x) ** 2)
            adv_loss = torch.mean((y_hat_adv - batch_y) ** 2)
            reg = params["lambda_reg"] * (torch.norm(W) ** 2 + torch.norm(A) ** 2 + torch.norm(b) ** 2)
            loss_ae = rec_loss - params["eta_clf"] * adv_loss + reg
            loss_ae.backward()
            ae_opt.step()
            n_batches += 1

        obs, rec = compute_measured_observables(ae, lat_dis, X, y, U, v, lam, params["eta"], params["g"], device)
        measured.append(scalarize_observables(obs, rec))
        elapsed_time += lr * max(n_batches, 1)
        times.append(elapsed_time)

    return ae, lat_dis, np.asarray(times, dtype=float), measured, observables_to_state(initial_obs)


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
        ax.set_xlabel("time")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend()

    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        "Branch-B Theory vs Measured Linear Fader Dynamics\n"
        f"noise_total={params['noise_total']}, lambda_reg={params['lambda_reg']}, "
        f"lam_sig={params['lam_sig']}, lambda_C={params['lambda_C']}, "
        f"alpha_C={params['alpha_C']}, eta_clf={params['eta_clf']}, h_scale={params['h_scale']}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_file, dpi=160, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Compare Branch-B ODE against trained linear Fader observables.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher_seed", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=TRAIN_CONFIG["n_epochs"])
    parser.add_argument("--n_samples", type=int, default=TRAIN_CONFIG["n_samples"])
    parser.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--learning_rate", type=float, default=TRAIN_CONFIG["learning_rate"])
    parser.add_argument("--theory_t_final", type=float, default=THEORY_T_FINAL)
    parser.add_argument("--theory_points", type=int, default=THEORY_POINTS)
    parser.add_argument("--noise_total", type=float, default=FIXED_VALUES["noise_total"])
    parser.add_argument("--lambda_reg", type=float, default=FIXED_VALUES["lambda_reg"])
    parser.add_argument("--lam_sig", type=float, default=FIXED_VALUES["lam_sig"])
    parser.add_argument("--lambda_C", type=float, default=FIXED_VALUES["lambda_C"])
    parser.add_argument("--alpha_C", type=float, default=FIXED_VALUES["alpha_C"])
    parser.add_argument("--eta_clf", type=float, default=FIXED_VALUES["eta_clf"])
    parser.add_argument("--h_scale", type=float, default=FIXED_VALUES["h_scale"])
    parser.add_argument("--out_plot", type=str, default=OUT_PLOT)
    parser.add_argument("--out_data", type=str, default=OUT_DATA)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    params = build_params(
        args.noise_total,
        args.lambda_reg,
        args.lam_sig,
        args.lambda_C,
        args.alpha_C,
        args.eta_clf,
        args.h_scale,
    )

    X, y, U, v, lam = generate_dataset(
        args.n_samples,
        N_DIM,
        R_DIM,
        params["lam_sig"],
        params["noise_total"],
        params["h_scale"],
        seed=args.teacher_seed,
    )

    _, _, times, measured_list, x0 = train_and_measure(
        X,
        y,
        U,
        v,
        lam,
        params,
        n_epochs=args.n_epochs,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        device=device,
    )

    theory_t_final = max(float(args.theory_t_final), float(times[-1]))
    theory_times_dense = np.linspace(0.0, theory_t_final, int(args.theory_points))
    theory_states_dense = integrate_theory(x0, params, theory_times_dense)
    theory_list_dense = [scalarize_state(x, params) for x in theory_states_dense]

    keys = list(measured_list[0].keys())
    measured_hist = {k: np.array([row[k] for row in measured_list], dtype=float) for k in keys}
    theory_hist_dense = {k: np.array([row[k] for row in theory_list_dense], dtype=float) for k in keys}

    # Resample long-horizon theory onto measured checkpoints for direct overlay.
    theory_hist = {
        k: np.interp(times, theory_times_dense, theory_hist_dense[k]) for k in keys
    }

    plot_comparison(times, theory_times_dense, theory_hist_dense, measured_hist, params, args.out_plot)

    np.savez(
        args.out_data,
        times=times,
        theory_times_dense=theory_times_dense,
        **{f"measured_{k}": v for k, v in measured_hist.items()},
        **{f"theory_{k}": v for k, v in theory_hist.items()},
        **{f"theory_dense_{k}": v for k, v in theory_hist_dense.items()},
        noise_total=params["noise_total"],
        lambda_reg=params["lambda_reg"],
        lam_sig=params["lam_sig"],
        lambda_C=params["lambda_C"],
        alpha_C=params["alpha_C"],
        eta_clf=params["eta_clf"],
        h_scale=params["h_scale"],
        eta=params["eta"],
        g=params["g"],
    )
    print(f"Saved -> {args.out_plot}")
    print(f"Saved -> {args.out_data}")


if __name__ == "__main__":
    main()
