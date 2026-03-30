#!/usr/bin/env python3
"""
Branch B Phase Diagram Sweeps - matching 3d_sim.py structure exactly

Performs 2D and 1D parameter sweeps and generates phase diagrams.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")
from scipy.integrate import solve_ivp

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

D_DIM = 3
R_DIM = 3

SWEEP_X = "lambda_C"
SWEEP_Y = "eta_clf"

SWEEP_RANGES = {
    "lambda_reg":  (0.001, 0.5, 20),
    "lambda_C":    (0.001, 2.0, 20),
    "eta_clf":     (0.001, 10.0, 20),
}

FIXED_VALUES = {
    "lambda_reg":  0.01,
    "lambda_C":    0.01,
    "eta_clf":     1.0,
}

# Teacher parameters
TEACHER_PARAMS = {
    "lam_sig": 1.0,
    "sigma2_y": 0.1,
    "eta": 0.1,
}

T_FINAL = 100.0
RTOL = 1e-5
ATOL = 1e-7

TOL = dict(
    tol_zero=0.15,
    tol_one=0.15,
    tol_nonzero=0.15,
    tol_q=1e-8,
    tol_beta_soft=0.15,
    tol_entangle=0.08,
    tol_rand_rel=0.03,
    tol_improve=0.05,
    tol_M_learn=0.7,
    tol_N_learn=0.7,
)

OUT_2D = "phase_diagram_2d_branch_b.png"
OUT_1D = "phase_diagram_1d_branch_b.png"
OUT_TRAJ_PREFIX = "trajectory_branch_b"

# ─────────────────────────────────────────────────────────────────────────────
# PACK / UNPACK
# ─────────────────────────────────────────────────────────────────────────────

MAT_DD = D_DIM * D_DIM
MAT_RD = R_DIM * D_DIM
VEC_D = D_DIM
VEC_R = R_DIM

def pack_state(M, s, N, a, beta, rho, C, Q, T, u, t, B, m):
    return np.concatenate([
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
    ])

def unpack_state(X):
    idx = 0
    M = X[idx:idx + D_DIM * R_DIM].reshape(D_DIM, R_DIM); idx += D_DIM * R_DIM
    s = X[idx:idx + VEC_D]; idx += VEC_D
    N = X[idx:idx + R_DIM * D_DIM].reshape(R_DIM, D_DIM); idx += R_DIM * D_DIM
    a = X[idx:idx + VEC_D]; idx += VEC_D
    beta = X[idx:idx + VEC_R]; idx += VEC_R
    rho = X[idx]; idx += 1
    C = X[idx:idx + VEC_D]; idx += VEC_D
    Q = X[idx:idx + MAT_DD].reshape(D_DIM, D_DIM); idx += MAT_DD
    T = X[idx:idx + MAT_DD].reshape(D_DIM, D_DIM); idx += MAT_DD
    u = X[idx:idx + VEC_D]; idx += VEC_D
    t = X[idx:idx + VEC_D]; idx += VEC_D
    B = X[idx:idx + MAT_DD].reshape(D_DIM, D_DIM); idx += MAT_DD
    m = X[idx]; idx += 1
    return M, s, N, a, beta, rho, C, Q, T, u, t, B, m

# ─────────────────────────────────────────────────────────────────────────────
# INITIAL CONDITION
# ─────────────────────────────────────────────────────────────────────────────

def make_initial_state():
    M = 0.10 * np.eye(D_DIM, R_DIM)
    N = 0.10 * np.eye(R_DIM, D_DIM)
    Q = 0.09 * np.eye(D_DIM)
    T = 0.09 * np.eye(D_DIM)
    B = 0.09 * np.eye(D_DIM)
    s = np.array([0.05, -0.03, 0.02], dtype=float)
    a = np.array([0.04, 0.01, -0.02], dtype=float)
    beta = np.array([0.02, 0.00, -0.01], dtype=float)
    C = np.array([0.01, 0.00, 0.00], dtype=float)
    u = np.zeros(D_DIM, dtype=float)
    t = np.zeros(D_DIM, dtype=float)
    rho = 0.9
    m = 0.08
    return pack_state(M, s, N, a, beta, rho, C, Q, T, u, t, B, m)

X0 = make_initial_state()
STATE_DIM = len(X0)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

ALL_PHASES = [
    "disentangled representation", "entangled representation",
    "label_loss", "no_learning", "other"
]

PHASE_COLOR = {
    "disentangled representation":  "#2ecc71",
    "entangled representation":     "#3498db",
    "label_loss":                   "#f39c12",
    "no_learning":                  "#e74c3c",
    "other":                        "#888888",
}

PHASE_LABEL = {
    "disentangled representation":  "Disentangled representation  (||N||/√trQ≈1, β≈0, s≈a≈0)",
    "entangled representation":     "Entangled representation  (||M||/√trQ≈1, ||N||/√trQ≈1, s>0)",
    "label_loss":                   "Label loss  (a≈β≈ρ≈0)",
    "no_learning":                  "No learning  (||M||/√trQ<1)",
    "other":                        "Other",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def random_reconstruction_loss(g, lam_sig, eta):
    """Reconstruction loss of random predictor."""
    return float(R_DIM * lam_sig + g + eta * D_DIM)

def sym(A):
    return 0.5 * (A + A.T)

def q_scale(Q):
    return np.sqrt(max(np.trace(sym(Q)), TOL["tol_q"]))

def normalized_overlaps(x):
    M, s, N, a, beta, rho, C, Q, T, u, t, B, m = unpack_state(x)
    scale = q_scale(Q)
    M_tilde = np.linalg.norm(M, ord="fro") / scale
    N_tilde = np.linalg.norm(N, ord="fro") / scale
    return M_tilde, N_tilde, scale

def classify(x, g, lam_sig, eta):
    """Classify phase."""
    tz = TOL["tol_zero"]
    M, s, N, a, beta, rho, C, Q, T, u, t, B, m = unpack_state(x)
    M_tilde, N_tilde, _ = normalized_overlaps(x)

    s_norm = np.linalg.norm(s)
    a_norm = np.linalg.norm(a)
    beta_norm = np.linalg.norm(beta)
    
    L_rec = reconstruction_error(x, g, lam_sig, eta)
    L_rand = random_reconstruction_loss(g, lam_sig, eta)
    rel_gap = abs(L_rec - L_rand) / max(L_rand, 1e-12)
    gain = (L_rand - L_rec) / max(L_rand, 1e-12)

    M_learned = M_tilde > TOL["tol_M_learn"]
    N_learned = N_tilde > TOL["tol_N_learn"]
    nuisance_active = s_norm > TOL["tol_beta_soft"]
    nuisance_small = s_norm < TOL["tol_beta_soft"]

    if rel_gap < TOL["tol_rand_rel"] and M_tilde < 0.5 and N_tilde < 0.5:
        return "no_learning"
    if a_norm < tz and beta_norm < tz and abs(s_norm) < tz:
        return "label_loss"
    if M_learned and N_learned and nuisance_small:
        return "disentangled representation"
    if M_learned and N_learned and nuisance_active:
        return "entangled representation"
    return "other"

def reconstruction_error(x, g, lam_sig, eta):
    """Effective reconstruction error."""
    M, s, N, a, beta, rho, C, Q, T, u, t, B, m = unpack_state(x)
    Lambda = lam_sig * np.eye(R_DIM)
    S = M @ Lambda @ M.T + g * np.outer(s, s) + eta * Q
    G = N.T @ Lambda @ M.T + g * np.outer(a, s) + eta * B
    trace_sigma_eff = np.trace(Lambda) + g + eta * D_DIM
    err = (trace_sigma_eff + np.trace(T @ S) - 2.0 * np.trace(G) 
           + 2.0 * g * (u @ s - rho) + g * m)
    return float(np.real(err))

# ─────────────────────────────────────────────────────────────────────────────
# RHS
# ─────────────────────────────────────────────────────────────────────────────

def rhs(tau, X, g, lam_sig, eta, lW, lA, lb, lC, alpha_C, eta_clf):
    M, s, N, a, beta, rho, C, Q, T, u, t, B, m = unpack_state(X)

    Lambda = lam_sig * np.eye(R_DIM)
    D = (lam_sig + eta) * np.eye(R_DIM)
    kappa = g + eta
    CCt = np.outer(C, C)

    S = M @ Lambda @ M.T + g * np.outer(s, s) + eta * Q
    G = N.T @ Lambda @ M.T + g * np.outer(a, s) + eta * B
    J = N.T @ Lambda @ N + g * np.outer(a, a) + eta * T
    H = M @ Lambda @ beta + g * rho * s + eta * t
    q = N.T @ Lambda @ beta + g * rho * a + eta * u

    X_aux = T @ S - G + g * np.outer(u, s)

    dM = -2 * (T @ M @ D - N.T @ D) + 2 * eta_clf * CCt @ M @ D - 2 * lW * M
    ds = -2 * (kappa * (T @ s) - kappa * a + g * u) + 2 * eta_clf * kappa * CCt @ s - 2 * eta_clf * g * C - 2 * lW * s
    dN = -2 * (N @ S - D @ M.T + g * np.outer(beta, s)) - 2 * lA * N
    da = -2 * (S @ a - kappa * s + g * rho * s) - 2 * lA * a
    dbeta = -2 * g * (N @ s + beta) - 2 * lb * beta
    drho = -2 * g * (a @ s - 1.0 + rho) - 2 * lb * rho
    dC = -2 * alpha_C * (eta_clf * (S @ C - g * s) + lC * C)
    dQ = -2 * X_aux - 2 * X_aux.T + 2 * eta_clf * (CCt @ S + S @ CCt) - 2 * eta_clf * g * (np.outer(C, s) + np.outer(s, C)) - 4 * lW * Q
    dT = -2 * X_aux - 2 * X_aux.T - 4 * lA * T
    du = -2 * (S @ u - H + g * m * s) - 2 * g * (T @ s - a + u) - 2 * (lA + lb) * u
    dt_ = -2 * (T @ H - q + g * rho * u) + 2 * eta_clf * CCt @ H - 2 * eta_clf * g * rho * C - 2 * g * (B.T @ s - s + t) - 2 * (lW + lb) * t
    dB = -2 * (S @ B - S + g * np.outer(s, t)) - 2 * (G @ T - J + g * np.outer(a, u)) + 2 * eta_clf * G @ CCt - 2 * eta_clf * g * np.outer(a, C) - 2 * (lA + lW) * B
    dm = -4 * g * (u @ s - rho + m) - 4 * lb * m

    return pack_state(dM, ds, dN, da, dbeta, drho, dC, dQ, dT, du, dt_, dB, dm)

# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def integrate(g, lam_sig, eta, lW, lA, lb, lC, alpha_C, eta_clf):
    sol = solve_ivp(
        rhs, (0, T_FINAL), X0, method="RK45",
        args=(g, lam_sig, eta, lW, lA, lb, lC, alpha_C, eta_clf),
        rtol=RTOL, atol=ATOL, dense_output=False
    )
    x = sol.y[:, -1] if sol.y.shape[1] > 0 else np.zeros(STATE_DIM)
    return x if np.all(np.isfinite(x)) else np.zeros(STATE_DIM)

# ─────────────────────────────────────────────────────────────────────────────
# SWEEPS
# ─────────────────────────────────────────────────────────────────────────────

def run_2d_sweep():
    xmin, xmax, nx = SWEEP_RANGES[SWEEP_X]
    ymin, ymax, ny = SWEEP_RANGES[SWEEP_Y]
    vals_x = np.linspace(xmin, xmax, nx)
    vals_y = np.linspace(ymin, ymax, ny)

    phase_grid = np.empty((ny, nx), dtype=object)
    M_grid = np.zeros((ny, nx))
    N_grid = np.zeros((ny, nx))
    rho_grid = np.zeros((ny, nx))
    rec_grid = np.zeros((ny, nx))

    g = TEACHER_PARAMS["sigma2_y"]
    lam_sig = TEACHER_PARAMS["lam_sig"]
    eta = TEACHER_PARAMS["eta"]

    print(f"\n2D sweep: {SWEEP_X} [{xmin:.3g}, {xmax:.3g}, {nx}pts] × {SWEEP_Y} [{ymin:.3g}, {ymax:.3g}, {ny}pts]")

    import time
    t0 = time.time()
    for j, yv in enumerate(vals_y):
        for i, xv in enumerate(vals_x):
            params_dict = dict(FIXED_VALUES)
            params_dict[SWEEP_X] = xv
            params_dict[SWEEP_Y] = yv

            lW = params_dict["lambda_reg"]
            lA = params_dict["lambda_reg"]
            lb = params_dict["lambda_reg"]
            lC = params_dict["lambda_C"]
            alpha_C = 1.0
            eta_clf = params_dict["eta_clf"]

            x = integrate(g, lam_sig, eta, lW, lA, lb, lC, alpha_C, eta_clf)

            M_tilde, N_tilde, _ = normalized_overlaps(x)
            phase_grid[j, i] = classify(x, g, lam_sig, eta)
            M_grid[j, i] = M_tilde
            N_grid[j, i] = N_tilde

            _, _, _, _, _, rho, _, _, _, _, _, _, _ = unpack_state(x)
            rho_grid[j, i] = rho
            rec_grid[j, i] = reconstruction_error(x, g, lam_sig, eta)

        elapsed = time.time() - t0
        done = (j + 1) * nx
        total = nx * ny
        eta_rem = elapsed / done * (total - done) if done else 0.0
        bar = "█" * ((j + 1) * 20 // ny) + "░" * (20 - (j + 1) * 20 // ny)
        print(f"  [{bar}] row {j + 1:3d}/{ny}  {elapsed:.1f}s  ~{eta_rem:.0f}s left", end="\r", flush=True)
    print()

    return vals_x, vals_y, phase_grid, M_grid, N_grid, rho_grid, rec_grid

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING (using 3d_sim.py style)
# ─────────────────────────────────────────────────────────────────────────────

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

def plot_2d(vals_x, vals_y, phase_grid, M_grid, N_grid, rho_grid, rec_grid):
    phase_int = np.vectorize(p2i)(phase_grid)
    present = [p for p in ALL_PHASES if np.any(phase_grid == p)]
    flat = phase_grid.ravel()
    total = len(flat)

    cmap_phase = mcolors.ListedColormap([PHASE_COLOR[p] for p in ALL_PHASES])
    norm_phase = mcolors.BoundaryNorm(np.arange(len(ALL_PHASES) + 1) - 0.5, cmap_phase.N)

    fig, axes = plt.subplots(1, 5, figsize=(27, 5.5))
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    ax.pcolormesh(vals_x, vals_y, phase_int, cmap=cmap_phase, norm=norm_phase, shading="nearest")
    sax(ax)
    ax.set_xlabel(SWEEP_X, color="white", fontsize=10.5)
    ax.set_ylabel(SWEEP_Y, color="white", fontsize=10.5)
    ax.set_title("Phase diagram", color="white", fontsize=13, pad=10)

    count_lines = [f"{PHASE_LABEL[p].split('(')[0].strip()}: {int(np.sum(flat == p))} ({100 * np.sum(flat == p) / total:.0f}%)"
                   for p in present]
    ax.text(0.98, 0.02, "\n".join(count_lines),
            transform=ax.transAxes, fontsize=7.5, color="white",
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc=PANEL, ec="#666", alpha=0.9))

    patches = [mpatches.Patch(color=PHASE_COLOR[p], label=PHASE_LABEL[p]) for p in present]
    ax.legend(handles=patches, fontsize=8, loc="upper right",
              facecolor=PANEL, edgecolor="#555", labelcolor="white", framealpha=0.92)

    for idx, (data, title, cmap) in enumerate([
        (M_grid, "||M|| / √tr(Q)", "plasma"),
        (N_grid, "||N|| / √tr(Q)", "cividis"),
        (rho_grid, "ρ  (label alignment)", "coolwarm"),
        (rec_grid, "Reconstruction error", "viridis")
    ], 1):
        ax = axes[idx]
        if title == "ρ  (label alignment)":
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

    fig.suptitle(f"Phase diagram: {SWEEP_X} × {SWEEP_Y}", color="white", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_2D, dpi=155, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {OUT_2D}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    valid = set(SWEEP_RANGES.keys())
    assert SWEEP_X in valid, f"SWEEP_X must be one of {valid}"
    assert SWEEP_Y in valid, f"SWEEP_Y must be one of {valid}"
    assert SWEEP_X != SWEEP_Y, "SWEEP_X and SWEEP_Y must be different"

    vals_x, vals_y, phase_grid, M_grid, N_grid, rho_grid, rec_grid = run_2d_sweep()
    plot_2d(vals_x, vals_y, phase_grid, M_grid, N_grid, rho_grid, rec_grid)

    print("\nDone.")
