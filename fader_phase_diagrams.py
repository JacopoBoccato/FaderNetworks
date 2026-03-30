#!/usr/bin/env python3
"""
Branch B Validation: Train FaderNetwork & Compute Observables for Phase Diagrams

For each configuration:
1. Train FaderNetwork on synthetic teacher data
2. Extract trained weights W, A, b, C
3. Compute Branch B observables from trained system
4. Classify phase and collect metrics
5. Generate phase diagrams matching 3d_sim.py layout
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

from src.branch_b_observables import compute_branch_b_observables

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

D_DIM = 3  # Latent dimension
R_DIM = 3  # Signal rank
N_DIM = 32  # Data dimension (matching macroscopic theory)

SWEEP_X = "lam_sig"  # Teacher eigenvalue strength (isotropic)
SWEEP_Y = "eta_clf"   # Classifier loss strength

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
    "h_scale":     1.0,
}

# Noise splitting: eta_fraction * noise_total = eta (reconstruction), (1-eta_fraction) = g (label)
ETA_FRACTION = 0.5
H_DIRECTION = np.array([0.5, 0.5, 0.0], dtype=float)
H_DIRECTION = H_DIRECTION / max(np.linalg.norm(H_DIRECTION), 1e-12)

# Training parameters
TRAIN_CONFIG = {
    "n_epochs": 1000,
    "batch_size": 32,
    "epoch_size": 512,
    "n_samples": 50000,
    "n_valid": 64,
    "learning_rate": 0.0001,
}

CONVERGENCE_CONFIG = {
    "window": 10,              # Number of epochs used to assess stationarity
    "min_epochs": 40,          # Minimum epochs before declaring convergence
    "rel_improve_tol": 0.02,   # Relative change between consecutive windows
    "stability_tol": 0.05,     # Relative std in the latest window
}

OUT_2D = "phase_diagram_fader_trained.png"
OUT_DATA = "phase_diagram_data.npz"

# ─────────────────────────────────────────────────────────────────────────────
# TEACHER MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_teacher(n, r, h_scale, seed=0):
    """Build U, v with controllable non-orthogonality h = U^T v.
    
    U: n×r orthonormal matrix (signal subspace)
    v: n-dimensional unit vector (label direction)
    h: r-dimensional vector, h = U^T v
    """
    assert D_DIM == R_DIM, "Current h construction assumes D_DIM == R_DIM."

    rng = np.random.default_rng(seed)
    
    # U is orthonormal; v_perp is orthogonal complement direction
    Q, _ = np.linalg.qr(rng.standard_normal((n, r + 1)))
    U = Q[:, :r]  # n×r orthonormal
    v_perp = Q[:, r]

    h = h_scale * H_DIRECTION
    h_norm2 = float(np.dot(h, h))
    if h_norm2 >= 1.0:
        h = h / max(np.linalg.norm(h), 1e-12) * 0.999
        h_norm2 = float(np.dot(h, h))

    # v = U h + sqrt(1-||h||^2) * v_perp guarantees U^T v = h and ||v||=1
    v = U @ h + np.sqrt(max(1.0 - h_norm2, 1e-12)) * v_perp
    v = v / max(np.linalg.norm(v), 1e-12)
    
    return U, v, h


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


def generate_dataset(n_samples, n, r, lam_sig, noise_total, h_scale, seed=0):
    """Generate X, y from teacher model."""
    p = build_params(noise_total, 0.0, lam_sig, 0.0, 1.0, 1.0, h_scale)
    eta = p["eta"]
    g = p["g"]

    U, v, h_vec = build_teacher(n, r, h_scale, seed)
    rng = np.random.default_rng(seed)
    
    c = rng.standard_normal((n_samples, r))
    y = np.sqrt(g) * rng.standard_normal((n_samples, 1))
    a = rng.standard_normal((n_samples, n))

    # Isotropic Λ = lam_sig * I_r; scale signal coefficients by sqrt(lam_sig)
    c_scaled = np.sqrt(max(lam_sig, 1e-12)) * c
    X = (U @ c_scaled.T).T + (v[:, None] @ y.T).T + np.sqrt(eta) * a
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.squeeze(), dtype=torch.float32).unsqueeze(-1)

    lam = lam_sig * np.ones(r, dtype=float)
    return X, y, U, v, lam, eta, g, h_vec


# ─────────────────────────────────────────────────────────────────────────────
# LINEAR FADER MODEL
# ─────────────────────────────────────────────────────────────────────────────

class LinearFader(nn.Module):
    """Linear Fader: encoder W, decoder A+b, classifier C."""
    def __init__(self, n_input, n_latent):
        super().__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        
        # Encoder: X -> z, z = W @ X
        self.W = nn.Parameter(torch.randn(n_latent, n_input) / np.sqrt(n_input))
        
        # Decoder: z -> X_hat, X_hat = A @ z + b
        self.A = nn.Parameter(torch.randn(n_input, n_latent) / np.sqrt(n_latent))
        self.b = nn.Parameter(torch.zeros(n_input))
        
        # Classifier: z -> y_hat, y_hat = C @ z
        self.C = nn.Parameter(torch.randn(n_latent, 1) / np.sqrt(n_latent))
    
    def encode(self, X):
        """X: [batch, n_input] -> z: [batch, n_latent]"""
        return torch.matmul(X, self.W.T)
    
    def decode(self, z):
        """z: [batch, n_latent] -> X_hat: [batch, n_input]"""
        return torch.matmul(z, self.A.T) + self.b.unsqueeze(0)
    
    def classify(self, z):
        """z: [batch, n_latent] -> y_hat: [batch, 1]"""
        return torch.matmul(z, self.C)
    
    def forward(self, X):
        z = self.encode(X)
        X_hat = self.decode(z)
        y_hat = self.classify(z)
        return X_hat, y_hat, z


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_fader(X_train, y_train, lambda_reg, lambda_C, alpha_C, eta_clf, config, device):
    """Train linear Fader and return model plus convergence diagnostics."""
    
    model = LinearFader(N_DIM, D_DIM).to(device)
    base_lr = config['learning_rate']
    optimizer = torch.optim.Adam([
        {"params": [model.W, model.A, model.b], "lr": base_lr},
        {"params": [model.C], "lr": base_lr * alpha_C},
    ])
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    
    model.train()
    
    n_batches = config['epoch_size'] // config['batch_size']
    epoch_losses = []
    
    for epoch in range(config['n_epochs']):
        loss_acc = 0.0
        for batch_idx in range(n_batches):
            # Sample batch
            idx = np.random.choice(len(X_train), config['batch_size'], replace=False)
            batch_x = X_train[idx]
            batch_y = y_train[idx]
            
            optimizer.zero_grad()
            
            # Forward pass
            X_hat, y_hat, z = model(batch_x)
            
            # Reconstruction loss
            rec_loss = torch.mean((X_hat - batch_x) ** 2)
            
            # Classifier loss (adversarial: minimize classification accuracy)
            clf_loss = torch.mean((y_hat - batch_y) ** 2)
            
            # Regularization aligned with 3d_sim controls
            reg_wa = lambda_reg * (torch.norm(model.W) ** 2 + torch.norm(model.A) ** 2 + torch.norm(model.b) ** 2)
            reg_c = lambda_C * (torch.norm(model.C) ** 2)
            
            # Total loss
            loss = rec_loss + eta_clf * clf_loss + reg_wa + reg_c
            
            loss.backward()
            optimizer.step()
            loss_acc += float(loss.item())

        epoch_losses.append(loss_acc / max(n_batches, 1))
    
    # Convergence from loss stationarity across two consecutive windows.
    win = CONVERGENCE_CONFIG["window"]
    min_epochs = CONVERGENCE_CONFIG["min_epochs"]
    rel_tol = CONVERGENCE_CONFIG["rel_improve_tol"]
    stab_tol = CONVERGENCE_CONFIG["stability_tol"]

    converged = False
    rel_improve = np.inf
    stability = np.inf
    if len(epoch_losses) >= max(2 * win, min_epochs):
        prev = np.array(epoch_losses[-2 * win:-win], dtype=float)
        last = np.array(epoch_losses[-win:], dtype=float)
        prev_mean = float(np.mean(prev))
        last_mean = float(np.mean(last))
        rel_improve = abs(prev_mean - last_mean) / max(abs(prev_mean), 1e-12)
        stability = float(np.std(last) / max(abs(last_mean), 1e-12))
        converged = (rel_improve < rel_tol) and (stability < stab_tol)

    convergence_info = {
        "converged": converged,
        "final_loss": float(epoch_losses[-1]) if epoch_losses else np.nan,
        "rel_improve": float(rel_improve),
        "stability": float(stability),
    }
    return model, convergence_info



def extract_observables(model, X_valid, y_valid, U, v, lam, g, eta, device):
    """Extract trained system and compute Branch B observables."""
    
    model.eval()
    
    with torch.no_grad():
        # Extract weights directly
        W = model.W.cpu().numpy()  # [D_DIM, N_DIM]
        A = model.A.cpu().numpy()  # [N_DIM, D_DIM]
        b = model.b.cpu().numpy()  # [N_DIM]
        C = model.C.cpu().numpy()  # [D_DIM, 1]
        C = C.squeeze()  # [D_DIM]
        
        # Convert to torch for observable computation
        W_t = torch.tensor(W, dtype=torch.float32, device=device)
        A_t = torch.tensor(A, dtype=torch.float32, device=device)
        b_t = torch.tensor(b, dtype=torch.float32, device=device)
        C_t = torch.tensor(C, dtype=torch.float32, device=device)
        U_t = torch.tensor(U, dtype=torch.float32, device=device)
        v_t = torch.tensor(v, dtype=torch.float32, device=device)
        Lambda_t = torch.tensor(np.diag(lam), dtype=torch.float32, device=device)
        
        # Compute observables
        obs = compute_branch_b_observables(W_t, A_t, b_t, C_t, U_t, v_t, Lambda_t, eta, g)
        
        # Get norms
        M_norm = np.linalg.norm(obs['M'].cpu().numpy(), ord='fro')
        N_norm = np.linalg.norm(obs['N'].cpu().numpy(), ord='fro')
        Q_norm = np.sqrt(np.trace(obs['Q'].cpu().numpy()))
        
        M_tilde = M_norm / max(Q_norm, 1e-8)
        N_tilde = N_norm / max(Q_norm, 1e-8)
        rho = obs['rho'].item()
        s_norm = float(torch.norm(obs['s']).item())
        a_norm = float(torch.norm(obs['a']).item())
        beta_norm = float(torch.norm(obs['beta']).item())
        
        # Reconstruction error
        X_valid_t = X_valid.to(device)
        y_valid_t = y_valid.to(device)
        X_hat, _, _ = model(X_valid_t)
        rec_loss = torch.mean((X_hat - X_valid_t) ** 2).item()
    
    return M_tilde, N_tilde, rho, s_norm, a_norm, beta_norm, rec_loss



# ─────────────────────────────────────────────────────────────────────────────
# PHASE CLASSIFICATION
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
    "disentangled representation":  "Disentangled (||N||≈1, s≈a≈0)",
    "entangled representation":     "Entangled (||M||≈1, ||N||≈1, s>0)",
    "label_loss":                   "Label loss (a≈β≈0)",
    "no_learning":                  "No learning (||M||<0.5)",
    "other":                        "Other",
}

TOL = dict(
    tol_zero=0.15,
    tol_M_learn=0.7,
    tol_N_learn=0.7,
    tol_beta_soft=0.15,
)


def classify(M_tilde, N_tilde, rho, s_norm, a_norm, beta_norm):
    """Classify phase based on observables."""
    tz = TOL["tol_zero"]
    M_learned = M_tilde > TOL["tol_M_learn"]
    N_learned = N_tilde > TOL["tol_N_learn"]
    nuisance_active = s_norm > TOL["tol_beta_soft"]
    nuisance_small = s_norm < TOL["tol_beta_soft"]
    
    if M_tilde < 0.5 and N_tilde < 0.5:
        return "no_learning"
    if a_norm < tz and beta_norm < tz:
        return "label_loss"
    if M_learned and N_learned and nuisance_small:
        return "disentangled representation"
    if M_learned and N_learned and nuisance_active:
        return "entangled representation"
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# SWEEPS
# ─────────────────────────────────────────────────────────────────────────────

def run_2d_sweep():
    """Run 2D parameter sweep with linear Fader training."""
    
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
    
    device = torch.device("cpu")
    
    print(f"\n2D sweep: {SWEEP_X} [{xmin:.3g}, {xmax:.3g}, {nx}pts] × {SWEEP_Y} [{ymin:.3g}, {ymax:.3g}, {ny}pts]")
    print(f"Training on {TRAIN_CONFIG['n_samples']} samples, {TRAIN_CONFIG['n_epochs']} epochs per config\n")
    
    import time
    t0 = time.time()
    for j, yv in enumerate(vals_y):
        for i, xv in enumerate(vals_x):
            # Build parameter dict
            params_dict = dict(FIXED_VALUES)
            params_dict[SWEEP_X] = xv
            params_dict[SWEEP_Y] = yv

            p = build_params(
                params_dict["noise_total"],
                params_dict["lambda_reg"],
                params_dict["lam_sig"],
                params_dict["lambda_C"],
                params_dict["alpha_C"],
                params_dict["eta_clf"],
                params_dict["h_scale"],
            )
            
            # Generate teacher data for this configuration
            X_train, y_train, U, v, lam, eta, g, _ = generate_dataset(
                TRAIN_CONFIG['n_samples'], N_DIM, R_DIM, p["lam_sig"], p["noise_total"], p["h_scale"], seed=42
            )
            X_valid, y_valid, _, _, _, _, _, _ = generate_dataset(
                TRAIN_CONFIG['n_valid'], N_DIM, R_DIM, p["lam_sig"], p["noise_total"], p["h_scale"], seed=43
            )
            
            # Train linear Fader
            model, conv = train_fader(
                X_train, y_train,
                lambda_reg=p["lambda_reg"],
                lambda_C=p["lambda_C"],
                alpha_C=p["alpha_C"],
                eta_clf=p["eta_clf"],
                config=TRAIN_CONFIG,
                device=device,
            )
            
            # Extract observables
            M_tilde, N_tilde, rho, s_norm, a_norm, beta_norm, rec_loss = extract_observables(
                model, X_valid, y_valid, U, v, lam, g, eta, device
            )
            
            phase = classify(M_tilde, N_tilde, rho, s_norm, a_norm, beta_norm)
            
            phase_grid[j, i] = phase
            M_grid[j, i] = M_tilde
            N_grid[j, i] = N_tilde
            rho_grid[j, i] = rho
            rec_grid[j, i] = rec_loss
            converged_grid[j, i] = 1.0 if conv["converged"] else 0.0
            rel_improve_grid[j, i] = conv["rel_improve"]
            stability_grid[j, i] = conv["stability"]
        
        elapsed = time.time() - t0
        done = (j + 1) * nx
        total = nx * ny
        eta_rem = elapsed / done * (total - done) if done else 0.0
        bar = "█" * ((j + 1) * 20 // ny) + "░" * (20 - (j + 1) * 20 // ny)
        print(f"  [{bar}] row {j + 1:3d}/{ny}  {elapsed:.1f}s  ~{eta_rem:.0f}s left", end="\r", flush=True)
    print()
    
    conv_rate = 100.0 * np.mean(converged_grid)
    print(f"Converged configs: {int(np.sum(converged_grid))}/{converged_grid.size} ({conv_rate:.1f}%)")

    return (
        vals_x, vals_y, phase_grid, M_grid, N_grid, rho_grid, rec_grid,
        converged_grid, rel_improve_grid, stability_grid
    )


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
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


def plot_2d(vals_x, vals_y, phase_grid, M_grid, N_grid, rho_grid, rec_grid, converged_grid, rel_improve_grid, stability_grid):
    """Generate 2D phase diagram matching 3d_sim.py."""
    
    phase_int = np.vectorize(p2i)(phase_grid)
    present = [p for p in ALL_PHASES if np.any(phase_grid == p)]
    flat = phase_grid.ravel()
    total = len(flat)

    cmap_phase = mcolors.ListedColormap([PHASE_COLOR[p] for p in ALL_PHASES])
    norm_phase = mcolors.BoundaryNorm(np.arange(len(ALL_PHASES) + 1) - 0.5, cmap_phase.N)

    fig, axes = plt.subplots(1, 6, figsize=(32, 5.5))
    fig.patch.set_facecolor(BG)

    # Phase diagram
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

    # Observable grids
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

    # Convergence map
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

    fig.suptitle(f"Trained FaderNetwork Phase Diagram: {SWEEP_X} × {SWEEP_Y}",
                 color="white", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_2D, dpi=155, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {OUT_2D}")
    
    # Save data for later analysis
    np.savez(
        OUT_DATA,
        vals_x=vals_x, vals_y=vals_y,
        M_grid=M_grid, N_grid=N_grid, rho_grid=rho_grid, rec_grid=rec_grid,
        converged_grid=converged_grid, rel_improve_grid=rel_improve_grid, stability_grid=stability_grid,
        conv_window=CONVERGENCE_CONFIG["window"],
        conv_min_epochs=CONVERGENCE_CONFIG["min_epochs"],
        conv_rel_improve_tol=CONVERGENCE_CONFIG["rel_improve_tol"],
        conv_stability_tol=CONVERGENCE_CONFIG["stability_tol"],
    )
    print(f"Saved → {OUT_DATA}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    valid = set(SWEEP_RANGES.keys())
    assert SWEEP_X in valid, f"SWEEP_X must be one of {valid}"
    assert SWEEP_Y in valid, f"SWEEP_Y must be one of {valid}"
    assert SWEEP_X != SWEEP_Y, "SWEEP_X and SWEEP_Y must be different"
    
    vals_x, vals_y, phase_grid, M_grid, N_grid, rho_grid, rec_grid, converged_grid, rel_improve_grid, stability_grid = run_2d_sweep()
    plot_2d(vals_x, vals_y, phase_grid, M_grid, N_grid, rho_grid, rec_grid, converged_grid, rel_improve_grid, stability_grid)
    
    print("\nDone!")
