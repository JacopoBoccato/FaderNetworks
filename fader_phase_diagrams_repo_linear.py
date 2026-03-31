#!/usr/bin/env python3
"""
Phase diagrams using repository Fader structure in strict linear mode.

For each (x, y) point in the 2D sweep:
1. Build synthetic teacher data
2. Train one linear Fader implemented with src.model.AutoEncoder + LatentDiscriminator
3. Extract W, A, b, C from trained linear layers
4. Map to Branch-B observables
5. Plot phase / observables / convergence
"""

import argparse
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from src.model import AutoEncoder, LatentDiscriminator
from src.branch_b_observables import compute_branch_b_observables


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

D_DIM = 3
R_DIM = 3
N_DIM = 32

SWEEP_X = "lam_sig"
SWEEP_Y = "eta_clf"

SWEEP_RANGES = {
    "noise_total": (0.05, 6.0, 15),
    "lambda_reg":  (0.01, 2.0, 15),
    "lam_sig":     (0.2, 3.0, 15),
    "lambda_C":    (0.001, 2.0, 15),
    "alpha_C":     (1.0, 100.0, 15),
    "eta_clf":     (0.005, 50.0, 15),
    "h_scale":     (-0.95, 0.95, 15),
}

FIXED_VALUES = {
    "noise_total": 2.0,
    "lambda_reg":  0.1,
    "lam_sig":     1.0,
    "lambda_C":    0.01,
    "alpha_C":     1.0,
    "eta_clf":     1.0,
    "h_scale":     0.5,
}

ETA_FRACTION = 0.5
H_DIRECTION = np.array([0.5, 0.5, 0.0], dtype=float)
H_DIRECTION = H_DIRECTION / max(np.linalg.norm(H_DIRECTION), 1e-12)

TRAIN_CONFIG = {
    "n_epochs": 200,
    "max_epochs": 500,
    "batch_size": 64,
    "epoch_size": 1024,
    "n_samples": 4096,
    "n_valid": 256,
    "learning_rate": 0.001,
}

CONVERGENCE_CONFIG = {
    "window": 10,
    "min_epochs": 40,
    "rel_improve_tol": 0.02,
    "stability_tol": 0.05,
    "check_every": 1,
}

OUT_2D = "phase_diagram_repo_linear_fader.png"
OUT_DATA = "phase_diagram_repo_linear_data.npz"


# -----------------------------------------------------------------------------
# TEACHER / DATA
# -----------------------------------------------------------------------------

def build_params(noise_total, lambda_reg, lam_sig, lambda_C, alpha_C, eta_clf, h_scale, f_eta=ETA_FRACTION):
    eta = max(f_eta * noise_total, 1e-6)
    g = max((1.0 - f_eta) * noise_total, 1e-6)
    return dict(
        noise_total=noise_total,
        lambda_reg=lambda_reg,
        lam_sig=lam_sig,
        lambda_C=lambda_C,
        alpha_C=max(alpha_C, 1e-6),
        eta_clf=max(eta_clf, 1e-6),
        h_scale=h_scale,
        eta=eta,
        g=g,
    )


def build_teacher(n, r, h_scale, seed=0):
    rng = np.random.default_rng(seed)

    Q, _ = np.linalg.qr(rng.standard_normal((n, r + 1)))
    U = Q[:, :r]
    v_perp = Q[:, r]

    h = h_scale * H_DIRECTION
    h_norm2 = float(np.dot(h, h))
    if h_norm2 >= 1.0:
        h = h / max(np.linalg.norm(h), 1e-12) * 0.999
        h_norm2 = float(np.dot(h, h))

    v = U @ h + np.sqrt(max(1.0 - h_norm2, 1e-12)) * v_perp
    v = v / max(np.linalg.norm(v), 1e-12)

    return U, v, h


def generate_dataset(n_samples, n, r, lam_sig, noise_total, h_scale, seed=0):
    p = build_params(noise_total, 0.0, lam_sig, 0.0, 1.0, 1.0, h_scale)
    eta = p["eta"]
    g = p["g"]

    U, v, h_vec = build_teacher(n, r, h_scale, seed)
    rng = np.random.default_rng(seed)

    c = rng.standard_normal((n_samples, r))
    y = np.sqrt(g) * rng.standard_normal((n_samples, 1))
    a = rng.standard_normal((n_samples, n))

    c_scaled = np.sqrt(max(lam_sig, 1e-12)) * c
    X = (U @ c_scaled.T).T + (v[:, None] @ y.T).T + np.sqrt(eta) * a

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.squeeze(), dtype=torch.float32).unsqueeze(-1)

    lam = lam_sig * np.ones(r, dtype=float)
    return X, y, U, v, lam, eta, g, h_vec


# -----------------------------------------------------------------------------
# REPO LINEAR FADER
# -----------------------------------------------------------------------------

def make_repo_linear_fader(device):
    """Instantiate repository models in strict linear mode."""
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

    # Force exact linear structure for theory mapping:
    # z = W x, x_hat = [A b] [z;y], y_hat = C z
    enc_linear = [m for m in ae.encoder.modules() if isinstance(m, nn.Linear)]
    dec_linear = [m for m in ae.decoder.modules() if isinstance(m, nn.Linear)]
    clf_linear = [m for m in lat_dis.net.modules() if isinstance(m, nn.Linear)]

    assert len(enc_linear) == 1, "Encoder is not single-linear as expected."
    assert len(dec_linear) == 1, "Decoder is not single-linear as expected."
    assert len(clf_linear) == 1, "Classifier is not single-linear as expected."

    # Remove additive biases to keep exact structure.
    enc_linear[0].bias.data.zero_()
    dec_linear[0].bias.data.zero_()
    clf_linear[0].bias.data.zero_()
    enc_linear[0].bias.requires_grad_(False)
    dec_linear[0].bias.requires_grad_(False)
    clf_linear[0].bias.requires_grad_(False)

    return ae, lat_dis


def extract_linear_matrices(ae, lat_dis, device):
    """Extract W, A, b, C from repository linear modules."""
    enc = [m for m in ae.encoder.modules() if isinstance(m, nn.Linear)][0]
    dec = [m for m in ae.decoder.modules() if isinstance(m, nn.Linear)][0]
    clf = [m for m in lat_dis.net.modules() if isinstance(m, nn.Linear)][0]

    # Encoder: z = W x
    W = enc.weight

    # Decoder: x_hat = [A b] [z;y]
    dec_w = dec.weight
    A = dec_w[:, :D_DIM]
    b = dec_w[:, D_DIM]

    # Classifier: y_hat = C z
    C = clf.weight.view(-1)

    return (
        W.to(device),
        A.to(device),
        b.to(device),
        C.to(device),
    )


def train_repo_linear_fader(X_train, y_train, lambda_reg, lambda_C, alpha_C, eta_clf, config, device, conv_config=None):
    ae, lat_dis = make_repo_linear_fader(device)

    ae_params = [p for p in ae.parameters() if p.requires_grad]
    clf_params = [p for p in lat_dis.parameters() if p.requires_grad]

    lr = config["learning_rate"]
    opt_ae = torch.optim.Adam(ae_params, lr=lr)
    opt_clf = torch.optim.Adam(clf_params, lr=lr * alpha_C)

    X_train = X_train.to(device)
    y_train = y_train.to(device)

    n_batches = config["epoch_size"] // config["batch_size"]
    epoch_losses = []

    target_epochs = int(config["n_epochs"])
    max_epochs = int(config.get("max_epochs", target_epochs))

    conv_cfg = conv_config if conv_config is not None else CONVERGENCE_CONFIG
    win = conv_cfg["window"]
    min_epochs = conv_cfg["min_epochs"]
    rel_tol = conv_cfg["rel_improve_tol"]
    stab_tol = conv_cfg["stability_tol"]
    check_every = max(int(conv_cfg.get("check_every", 1)), 1)

    converged = False
    rel_improve = np.inf
    stability = np.inf
    epochs_run = 0

    for epoch_idx in range(max_epochs):
        loss_acc = 0.0
        for _ in range(n_batches):
            idx = np.random.choice(len(X_train), config["batch_size"], replace=False)
            batch_x = X_train[idx]
            batch_y = y_train[idx]

            # 1) Classifier step: update C only
            opt_clf.zero_grad()
            with torch.no_grad():
                z_det = ae.encode(batch_x)[-1]
            y_hat_clf = lat_dis(z_det)
            clf_loss = torch.mean((y_hat_clf - batch_y) ** 2)
            _, _, _, C = extract_linear_matrices(ae, lat_dis, device)
            clf_reg = lambda_C * (torch.norm(C) ** 2)
            loss_clf = clf_loss + clf_reg
            loss_clf.backward()
            opt_clf.step()

            # 2) Fader step: update W, A, b only
            for p in lat_dis.parameters():
                p.requires_grad_(False)
            opt_ae.zero_grad()
            enc_outputs, dec_outputs = ae(batch_x, batch_y)
            z = enc_outputs[-1]
            x_hat = dec_outputs[-1]
            y_hat_adv = lat_dis(z)

            rec_loss = torch.mean((x_hat - batch_x) ** 2)
            adv_loss = torch.mean((y_hat_adv - batch_y) ** 2)

            W, A, b, _ = extract_linear_matrices(ae, lat_dis, device)
            reg_wa = lambda_reg * (torch.norm(W) ** 2 + torch.norm(A) ** 2 + torch.norm(b) ** 2)
            loss_ae = rec_loss - eta_clf * adv_loss + reg_wa
            loss_ae.backward()
            opt_ae.step()
            for p in lat_dis.parameters():
                p.requires_grad_(True)

            loss_acc += float(loss_ae.item())

        epoch_losses.append(loss_acc / max(n_batches, 1))
        epochs_run = epoch_idx + 1

        ready_for_conv = epochs_run >= max(2 * win, min_epochs)
        should_check = (epoch_idx + 1) % check_every == 0
        if ready_for_conv and should_check:
            prev = np.array(epoch_losses[-2 * win:-win], dtype=float)
            last = np.array(epoch_losses[-win:], dtype=float)
            prev_mean = float(np.mean(prev))
            last_mean = float(np.mean(last))
            rel_improve = abs(prev_mean - last_mean) / max(abs(prev_mean), 1e-12)
            stability = float(np.std(last) / max(abs(last_mean), 1e-12))
            converged = (rel_improve < rel_tol) and (stability < stab_tol)
            if converged:
                break

    # Final convergence report if not already converged.
    if (not converged) and len(epoch_losses) >= max(2 * win, min_epochs):
        prev = np.array(epoch_losses[-2 * win:-win], dtype=float)
        last = np.array(epoch_losses[-win:], dtype=float)
        prev_mean = float(np.mean(prev))
        last_mean = float(np.mean(last))
        rel_improve = abs(prev_mean - last_mean) / max(abs(prev_mean), 1e-12)
        stability = float(np.std(last) / max(abs(last_mean), 1e-12))
        converged = (rel_improve < rel_tol) and (stability < stab_tol)

    conv = {
        "converged": converged,
        "final_loss": float(epoch_losses[-1]) if epoch_losses else np.nan,
        "rel_improve": float(rel_improve),
        "stability": float(stability),
        "epochs_run": int(epochs_run),
        "target_epochs": int(target_epochs),
        "max_epochs": int(max_epochs),
        "used_extension": bool(epochs_run > target_epochs),
        "hit_max_epochs": bool((epochs_run >= max_epochs) and (not converged)),
    }
    return ae, lat_dis, conv


# -----------------------------------------------------------------------------
# OBSERVABLES / PHASE
# -----------------------------------------------------------------------------

ALL_PHASES = [
    "disentangled representation", "entangled representation",
    "label_loss", "no_learning", "other"
]

PHASE_COLOR = {
    "disentangled representation": "#2ecc71",
    "entangled representation": "#3498db",
    "label_loss": "#f39c12",
    "no_learning": "#e74c3c",
    "other": "#888888",
}

PHASE_LABEL = {
    "disentangled representation": "Disentangled (||N||≈1, s≈a≈0)",
    "entangled representation": "Entangled (||M||≈1, ||N||≈1, s>0)",
    "label_loss": "Label loss (a≈β≈0)",
    "no_learning": "No learning (||M||<0.5)",
    "other": "Other",
}

TOL = {
    "tol_zero": 0.15,
    "tol_M_learn": 0.7,
    "tol_N_learn": 0.7,
    "tol_beta_soft": 0.15,
}


def classify(M_tilde, N_tilde, s_norm, a_norm, beta_norm):
    tz = TOL["tol_zero"]
    M_learned = M_tilde > TOL["tol_M_learn"]
    N_learned = N_tilde > TOL["tol_N_learn"]
    nuisance_active = s_norm > TOL["tol_beta_soft"]

    if M_tilde < 0.5 and N_tilde < 0.5:
        return "no_learning"
    if a_norm < tz and beta_norm < tz:
        return "label_loss"
    if M_learned and N_learned and not nuisance_active:
        return "disentangled representation"
    if M_learned and N_learned and nuisance_active:
        return "entangled representation"
    return "other"


def extract_observables_repo(ae, lat_dis, X_valid, y_valid, U, v, lam, g, eta, device):
    ae.eval()
    lat_dis.eval()
    with torch.no_grad():
        W, A, b, C = extract_linear_matrices(ae, lat_dis, device)

        U_t = torch.tensor(U, dtype=torch.float32, device=device)
        v_t = torch.tensor(v, dtype=torch.float32, device=device)
        Lambda_t = torch.tensor(np.diag(lam), dtype=torch.float32, device=device)

        obs = compute_branch_b_observables(W, A, b, C, U_t, v_t, Lambda_t, eta, g)

        M_norm = np.linalg.norm(obs["M"].cpu().numpy(), ord="fro")
        N_norm = np.linalg.norm(obs["N"].cpu().numpy(), ord="fro")
        Q_norm = np.sqrt(np.trace(obs["Q"].cpu().numpy()))

        M_tilde = M_norm / max(Q_norm, 1e-8)
        N_tilde = N_norm / max(Q_norm, 1e-8)
        rho = float(obs["rho"].item())

        s_norm = float(torch.norm(obs["s"]).item())
        a_norm = float(torch.norm(obs["a"]).item())
        beta_norm = float(torch.norm(obs["beta"]).item())

        X_valid_t = X_valid.to(device)
        y_valid_t = y_valid.to(device)
        x_hat = ae(X_valid_t, y_valid_t)[1][-1]
        rec_loss = torch.mean((x_hat - X_valid_t) ** 2).item()

    return M_tilde, N_tilde, rho, s_norm, a_norm, beta_norm, rec_loss


# -----------------------------------------------------------------------------
# SWEEP / PLOT
# -----------------------------------------------------------------------------

BG = "#1a1a2e"
PANEL = "#0f0f23"
GRID = "#2a2a4a"


def p2i(ph):
    return ALL_PHASES.index(ph) if ph in ALL_PHASES else len(ALL_PHASES) - 1


def sax(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.tick_params(colors="white", labelsize=9)
    ax.grid(True, color=GRID, alpha=0.7)


def _train_one_config(task):
    i, j, xv, yv, params_dict, train_config, conv_config, use_cuda, gpu_id = task

    # Isolate each worker to one GPU when CUDA is used.
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:0")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")

    p = build_params(
        params_dict["noise_total"],
        params_dict["lambda_reg"],
        params_dict["lam_sig"],
        params_dict["lambda_C"],
        params_dict["alpha_C"],
        params_dict["eta_clf"],
        params_dict["h_scale"],
    )

    seed_base = 1000 + j * 100 + i
    X_train, y_train, U, v, lam, eta, g, _ = generate_dataset(
        train_config["n_samples"], N_DIM, R_DIM,
        p["lam_sig"], p["noise_total"], p["h_scale"], seed=seed_base
    )
    X_valid, y_valid, _, _, _, _, _, _ = generate_dataset(
        train_config["n_valid"], N_DIM, R_DIM,
        p["lam_sig"], p["noise_total"], p["h_scale"], seed=seed_base + 1
    )

    ae, lat_dis, conv = train_repo_linear_fader(
        X_train, y_train,
        lambda_reg=p["lambda_reg"],
        lambda_C=p["lambda_C"],
        alpha_C=p["alpha_C"],
        eta_clf=p["eta_clf"],
        config=train_config,
        device=device,
        conv_config=conv_config,
    )

    M_tilde, N_tilde, rho, s_norm, a_norm, beta_norm, rec_loss = extract_observables_repo(
        ae, lat_dis, X_valid, y_valid, U, v, lam, g, eta, device
    )
    phase = classify(M_tilde, N_tilde, s_norm, a_norm, beta_norm)
    return dict(
        i=i,
        j=j,
        phase=phase,
        M_tilde=float(M_tilde),
        N_tilde=float(N_tilde),
        rho=float(rho),
        rec_loss=float(rec_loss),
        converged=bool(conv["converged"]),
        rel_improve=float(conv["rel_improve"]),
        stability=float(conv["stability"]),
        epochs_run=int(conv.get("epochs_run", train_config["n_epochs"])),
        used_extension=bool(conv.get("used_extension", False)),
        hit_max_epochs=bool(conv.get("hit_max_epochs", False)),
    )


def run_2d_sweep(device, num_workers=1, gpu_ids=None):
    xmin, xmax, nx = SWEEP_RANGES[SWEEP_X]
    ymin, ymax, ny = SWEEP_RANGES[SWEEP_Y]
    vals_x = np.linspace(xmin, xmax, nx)
    vals_y = np.linspace(ymin, ymax, ny)

    phase_grid = np.empty((ny, nx), dtype=object)
    M_grid = np.zeros((ny, nx))
    N_grid = np.zeros((ny, nx))
    rho_grid = np.zeros((ny, nx))
    rec_grid = np.zeros((ny, nx))
    converged_grid = np.zeros((ny, nx))
    rel_improve_grid = np.full((ny, nx), np.nan)
    stability_grid = np.full((ny, nx), np.nan)
    epochs_grid = np.zeros((ny, nx))

    total = nx * ny
    print(
        f"\n2D sweep: {SWEEP_X} [{xmin:.3g}, {xmax:.3g}, {nx}pts] "
        f"x {SWEEP_Y} [{ymin:.3g}, {ymax:.3g}, {ny}pts]"
    )
    print(
        f"Training: target_epochs={TRAIN_CONFIG['n_epochs']}, max_epochs={TRAIN_CONFIG['max_epochs']}, "
        f"min_epochs={CONVERGENCE_CONFIG['min_epochs']}, configs={total}"
    )

    use_cuda = (device.type == "cuda")
    gpu_ids = list(gpu_ids or [])
    if use_cuda and not gpu_ids:
        gpu_ids = list(range(torch.cuda.device_count()))
    if use_cuda and not gpu_ids:
        raise RuntimeError("CUDA requested but no GPUs are visible.")

    tasks = []
    for j, yv in enumerate(vals_y):
        for i, xv in enumerate(vals_x):
            params_dict = dict(FIXED_VALUES)
            params_dict[SWEEP_X] = float(xv)
            params_dict[SWEEP_Y] = float(yv)
            gpu_id = gpu_ids[(j * nx + i) % len(gpu_ids)] if use_cuda else -1
            tasks.append(
                (i, j, float(xv), float(yv), params_dict, dict(TRAIN_CONFIG), dict(CONVERGENCE_CONFIG), use_cuda, gpu_id)
            )

    import time
    t0 = time.time()
    done = 0
    ext_count = 0
    hit_max_count = 0
    workers = max(1, int(num_workers))

    mp_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
        futures = [ex.submit(_train_one_config, t) for t in tasks]
        for fut in as_completed(futures):
            res = fut.result()
            i = res["i"]
            j = res["j"]
            phase_grid[j, i] = res["phase"]
            M_grid[j, i] = res["M_tilde"]
            N_grid[j, i] = res["N_tilde"]
            rho_grid[j, i] = res["rho"]
            rec_grid[j, i] = res["rec_loss"]
            converged_grid[j, i] = 1.0 if res["converged"] else 0.0
            rel_improve_grid[j, i] = res["rel_improve"]
            stability_grid[j, i] = res["stability"]
            epochs_grid[j, i] = res["epochs_run"]
            ext_count += int(res["used_extension"])
            hit_max_count += int(res["hit_max_epochs"])

            done += 1
            elapsed = time.time() - t0
            eta_rem = elapsed / done * (total - done) if done else 0.0
            frac = done / max(total, 1)
            bar_fill = int(20 * frac)
            bar = "#" * bar_fill + "." * (20 - bar_fill)
            print(
                f"  [{bar}] {done:3d}/{total} done  {elapsed:.1f}s  ~{eta_rem:.0f}s left",
                end="\r",
                flush=True,
            )
    print()

    conv_rate = 100.0 * np.mean(converged_grid)
    print(f"Converged configs: {int(np.sum(converged_grid))}/{converged_grid.size} ({conv_rate:.1f}%)")
    print(f"Used extension beyond target_epochs: {ext_count}/{total}")
    print(f"Reached max_epochs without convergence: {hit_max_count}/{total}")

    return (
        vals_x, vals_y, phase_grid, M_grid, N_grid, rho_grid, rec_grid,
        converged_grid, rel_improve_grid, stability_grid, epochs_grid,
    )


def plot_2d(vals_x, vals_y, phase_grid, M_grid, N_grid, rho_grid, rec_grid, converged_grid):
    phase_int = np.vectorize(p2i)(phase_grid)
    present = [p for p in ALL_PHASES if np.any(phase_grid == p)]
    flat = phase_grid.ravel()
    total = len(flat)

    cmap_phase = mcolors.ListedColormap([PHASE_COLOR[p] for p in ALL_PHASES])
    norm_phase = mcolors.BoundaryNorm(np.arange(len(ALL_PHASES) + 1) - 0.5, cmap_phase.N)

    fig, axes = plt.subplots(1, 6, figsize=(32, 5.5))
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    ax.pcolormesh(vals_x, vals_y, phase_int, cmap=cmap_phase, norm=norm_phase, shading="nearest")
    sax(ax)
    ax.set_xlabel(SWEEP_X, color="white", fontsize=10.5)
    ax.set_ylabel(SWEEP_Y, color="white", fontsize=10.5)
    ax.set_title("Phase diagram", color="white", fontsize=13, pad=10)

    count_lines = [
        f"{PHASE_LABEL[p].split('(')[0].strip()}: {int(np.sum(flat == p))} ({100 * np.sum(flat == p) / total:.0f}%)"
        for p in present
    ]
    ax.text(
        0.98, 0.02, "\n".join(count_lines),
        transform=ax.transAxes, fontsize=7.5, color="white",
        va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc=PANEL, ec="#666", alpha=0.9),
    )

    patches = [mpatches.Patch(color=PHASE_COLOR[p], label=PHASE_LABEL[p]) for p in present]
    ax.legend(handles=patches, fontsize=8, loc="upper right",
              facecolor=PANEL, edgecolor="#555", labelcolor="white", framealpha=0.92)

    for idx, (data, title, cmap) in enumerate([
        (M_grid, "||M|| / sqrt(tr(Q))", "plasma"),
        (N_grid, "||N|| / sqrt(tr(Q))", "cividis"),
        (rho_grid, "rho (label alignment)", "coolwarm"),
        (rec_grid, "Reconstruction error", "viridis"),
    ], 1):
        ax = axes[idx]
        if title == "rho (label alignment)":
            im = ax.pcolormesh(vals_x, vals_y, data, cmap=cmap, vmin=0, vmax=1, shading="nearest")
        else:
            im = ax.pcolormesh(vals_x, vals_y, data, cmap=cmap, shading="nearest")
        sax(ax)
        ax.set_xlabel(SWEEP_X, color="white", fontsize=10.5)
        ax.set_ylabel(SWEEP_Y, color="white", fontsize=10.5)
        ax.set_title(title, color="white", fontsize=12)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
        cb.outline.set_edgecolor("#444")

    ax = axes[5]
    im = ax.pcolormesh(vals_x, vals_y, converged_grid, cmap="RdYlGn", vmin=0.0, vmax=1.0, shading="nearest")
    sax(ax)
    ax.set_xlabel(SWEEP_X, color="white", fontsize=10.5)
    ax.set_ylabel(SWEEP_Y, color="white", fontsize=10.5)
    ax.set_title("Convergence (0=no, 1=yes)", color="white", fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    cb.outline.set_edgecolor("#444")

    conv_rate = 100.0 * np.mean(converged_grid)
    ax.text(0.98, 0.02, f"Converged: {conv_rate:.1f}%",
            transform=ax.transAxes, fontsize=8, color="white",
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec="#666", alpha=0.9))

    fig.suptitle(f"Repo Linear Fader Phase Diagram: {SWEEP_X} x {SWEEP_Y}", color="white", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_2D, dpi=155, bbox_inches="tight", facecolor=BG)
    print(f"Saved -> {OUT_2D}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--workers", type=int, default=1, help="Parallel configs to run at once.")
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--fast", action="store_true", help="Use an aggressive fast profile.")
    parser.add_argument("--nx", type=int, default=None, help="Points on SWEEP_X.")
    parser.add_argument("--ny", type=int, default=None, help="Points on SWEEP_Y.")
    parser.add_argument("--n_epochs", type=int, default=None, help="Target epochs before extension.")
    parser.add_argument("--max_epochs", type=int, default=None, help="Hard cap if not converged.")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--n_valid", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epoch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--conv_window", type=int, default=None)
    parser.add_argument("--conv_min_epochs", type=int, default=None)
    parser.add_argument("--conv_rel_improve_tol", type=float, default=None)
    parser.add_argument("--conv_stability_tol", type=float, default=None)
    parser.add_argument("--conv_check_every", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    if args.fast:
        # Very fast sanity profile
        TRAIN_CONFIG["n_epochs"] = 30
        TRAIN_CONFIG["max_epochs"] = 80
        TRAIN_CONFIG["n_samples"] = 1024
        TRAIN_CONFIG["n_valid"] = 128
        TRAIN_CONFIG["batch_size"] = 64
        TRAIN_CONFIG["epoch_size"] = 512
        TRAIN_CONFIG["learning_rate"] = 0.001
        CONVERGENCE_CONFIG["min_epochs"] = 20
        CONVERGENCE_CONFIG["window"] = 6
        x0, x1, _ = SWEEP_RANGES[SWEEP_X]
        y0, y1, _ = SWEEP_RANGES[SWEEP_Y]
        SWEEP_RANGES[SWEEP_X] = (x0, x1, 6)
        SWEEP_RANGES[SWEEP_Y] = (y0, y1, 6)

    # Explicit CLI overrides
    if args.n_epochs is not None:
        TRAIN_CONFIG["n_epochs"] = args.n_epochs
    if args.max_epochs is not None:
        TRAIN_CONFIG["max_epochs"] = args.max_epochs
    if args.n_samples is not None:
        TRAIN_CONFIG["n_samples"] = args.n_samples
    if args.n_valid is not None:
        TRAIN_CONFIG["n_valid"] = args.n_valid
    if args.batch_size is not None:
        TRAIN_CONFIG["batch_size"] = args.batch_size
    if args.epoch_size is not None:
        TRAIN_CONFIG["epoch_size"] = args.epoch_size
    if args.learning_rate is not None:
        TRAIN_CONFIG["learning_rate"] = args.learning_rate
    if args.conv_window is not None:
        CONVERGENCE_CONFIG["window"] = args.conv_window
    if args.conv_min_epochs is not None:
        CONVERGENCE_CONFIG["min_epochs"] = args.conv_min_epochs
    if args.conv_rel_improve_tol is not None:
        CONVERGENCE_CONFIG["rel_improve_tol"] = args.conv_rel_improve_tol
    if args.conv_stability_tol is not None:
        CONVERGENCE_CONFIG["stability_tol"] = args.conv_stability_tol
    if args.conv_check_every is not None:
        CONVERGENCE_CONFIG["check_every"] = args.conv_check_every
    if args.nx is not None:
        x0, x1, _ = SWEEP_RANGES[SWEEP_X]
        SWEEP_RANGES[SWEEP_X] = (x0, x1, args.nx)
    if args.ny is not None:
        y0, y1, _ = SWEEP_RANGES[SWEEP_Y]
        SWEEP_RANGES[SWEEP_Y] = (y0, y1, args.ny)

    valid = set(SWEEP_RANGES.keys())
    assert SWEEP_X in valid and SWEEP_Y in valid and SWEEP_X != SWEEP_Y
    if TRAIN_CONFIG["max_epochs"] < TRAIN_CONFIG["n_epochs"]:
        raise ValueError("max_epochs must be >= n_epochs.")
    if CONVERGENCE_CONFIG["min_epochs"] > TRAIN_CONFIG["max_epochs"]:
        raise ValueError("conv_min_epochs must be <= max_epochs.")

    gpu_ids = []
    if args.gpus.strip():
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if device.type == "cuda" and not gpu_ids:
        gpu_ids = list(range(torch.cuda.device_count()))
    if device.type == "cuda" and not gpu_ids:
        raise RuntimeError("CUDA requested but no GPUs available.")
    if device.type == "cuda" and args.workers <= 1:
        workers = len(gpu_ids)
    else:
        workers = args.workers

    out = run_2d_sweep(device, num_workers=workers, gpu_ids=gpu_ids)
    vals_x, vals_y, phase_grid, M_grid, N_grid, rho_grid, rec_grid, converged_grid, rel_improve_grid, stability_grid, epochs_grid = out

    plot_2d(vals_x, vals_y, phase_grid, M_grid, N_grid, rho_grid, rec_grid, converged_grid)

    np.savez(
        OUT_DATA,
        vals_x=vals_x,
        vals_y=vals_y,
        M_grid=M_grid,
        N_grid=N_grid,
        rho_grid=rho_grid,
        rec_grid=rec_grid,
        converged_grid=converged_grid,
        rel_improve_grid=rel_improve_grid,
        stability_grid=stability_grid,
        epochs_grid=epochs_grid,
        conv_window=CONVERGENCE_CONFIG["window"],
        conv_min_epochs=CONVERGENCE_CONFIG["min_epochs"],
        conv_rel_improve_tol=CONVERGENCE_CONFIG["rel_improve_tol"],
        conv_stability_tol=CONVERGENCE_CONFIG["stability_tol"],
        conv_check_every=CONVERGENCE_CONFIG["check_every"],
        target_epochs=TRAIN_CONFIG["n_epochs"],
        max_epochs=TRAIN_CONFIG["max_epochs"],
    )
    print(f"Saved -> {OUT_DATA}")


if __name__ == "__main__":
    main()
