#!/usr/bin/env python3
"""
Run an ODE-only Branch-B parameter sweep and plot the final theory phase diagram.

This script mirrors the existing grid-sweep interface from
`fader_phase_diagrams_repo_linear.py`, but it:

1. fixes all parameters through CLI flags,
2. varies exactly two `ExperimentConfig` fields over a 2D grid,
3. builds the Branch-B initial observable state at each grid point, and
4. integrates only the ODE before classifying the final theory state with the
   same phase rules used by `plot_phase_diagram_from_npz.py`.
"""

import argparse
import os
from dataclasses import fields, replace
from typing import Any, Dict, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from phase_diagram import (
    ExperimentConfig,
    N_DIM,
    R_DIM,
    build_shared_params,
    compute_measured_observables,
    generate_dataset,
    integrate_theory,
    make_linear_fader,
    observables_to_state,
    scalarize_state,
)
from plot_phase_diagram_from_npz import (
    BG,
    CLASSIFIER_THRESHOLDS,
    DEFAULT_PHASE_COLORS,
    GRID,
    PANEL,
    classify_phase,
    ordered_phases,
    phase_to_int,
)


CONFIG_FIELDS = {field.name: field.type for field in fields(ExperimentConfig)}
DEFAULT_CONFIG = ExperimentConfig()
FINAL_METRIC_NAMES: Tuple[str, ...] = (
    "norm_M",
    "norm_s",
    "norm_N",
    "norm_a",
    "norm_beta",
    "rho",
    "norm_C",
    "norm_Q",
    "norm_T",
    "norm_u",
    "norm_t",
    "norm_B",
    "m",
    "reconstruction_error",
    "M_tilde",
    "N_tilde",
)

# Edit this block, then run:
#   python3 plot_ode_phase_diagram.py
#
# CLI flags still override these defaults when needed.
SCRIPT_SWEEP_DEFAULTS = {
    "vary_x": "lambda_reg",
    "vary_y": "eta_clf",
    "x_values": (0.1, 0.3, 0.5),
    "y_values": (0.5, 1.0, 1.5),
    "tau_max": 2.0,
    "out": os.path.join("results", "branch_b_ode_phase_diagram.png"),
    "out_data": os.path.join("results", "branch_b_ode_phase_diagram.npz"),
}

SCRIPT_CLASSIFIER_THRESHOLDS = {
    "loss_small_thresh": CLASSIFIER_THRESHOLDS["loss_small"],
    "small_thresh": CLASSIFIER_THRESHOLDS["small"],
    "active_thresh": CLASSIFIER_THRESHOLDS["active"],
    "rho_thresh": CLASSIFIER_THRESHOLDS["rho"],
}

SCRIPT_EXPERIMENT_DEFAULTS = replace(
    DEFAULT_CONFIG,
    noise_total=0.5,
    lambda_reg=0.5,
    lam_sig=15.0,
    lambda_C=0.1,
    alpha_C=1.0,
    eta_clf=1.0,
    gamma0=0.0,
    gamma_mu=0.0,
    h_scale=0.2,
    n_epochs=2000,
    n_samples=5000,
    batch_size=1000,
    epoch_size=0,
    learning_rate=5e-4,
    measure_every_batches=1,
    lambda_schedule=0,
    theory_points=1000,
    theory_solver="Radau",
    seed=0,
    teacher_seed=0,
    device="cpu",
)


def format_default_value_list(values: Sequence[Any]) -> str:
    return ",".join(str(value) for value in values)


def parse_scalar_value(raw: str, field_name: str) -> Any:
    if field_name not in CONFIG_FIELDS:
        valid = ", ".join(sorted(CONFIG_FIELDS))
        raise ValueError(f"Unknown config field '{field_name}'. Valid names: {valid}")

    field_type = CONFIG_FIELDS[field_name]
    if field_type is int:
        return int(raw)
    if field_type is float:
        return float(raw)
    if field_type is str:
        return str(raw)
    raise TypeError(f"Unsupported field type for '{field_name}': {field_type}")


def parse_value_list(raw: str, field_name: str) -> Tuple[Any, ...]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"No values provided for '{field_name}'.")
    return tuple(parse_scalar_value(item, field_name) for item in values)


def maybe_build_linspace(
    field_name: str,
    n_points: Optional[int],
    v_min: Optional[float],
    v_max: Optional[float],
) -> Optional[Tuple[Any, ...]]:
    if n_points is None or v_min is None or v_max is None:
        return None
    if n_points <= 0:
        raise ValueError("Grid size must be positive.")

    field_type = CONFIG_FIELDS[field_name]
    raw = np.linspace(v_min, v_max, int(n_points))
    if field_type is int:
        return tuple(int(round(x)) for x in raw)
    if field_type is float:
        return tuple(float(x) for x in raw)
    raise ValueError(f"Linspace grid is only supported for numeric fields, got '{field_name}'.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an ODE-only Branch-B phase-diagram sweep.")
    parser.add_argument("--vary_x", type=str, default=SCRIPT_SWEEP_DEFAULTS["vary_x"], choices=sorted(CONFIG_FIELDS))
    parser.add_argument("--vary_y", type=str, default=SCRIPT_SWEEP_DEFAULTS["vary_y"], choices=sorted(CONFIG_FIELDS))
    parser.add_argument(
        "--x_values",
        type=str,
        default=format_default_value_list(SCRIPT_SWEEP_DEFAULTS["x_values"]),
        help="Comma-separated values for the x-axis parameter.",
    )
    parser.add_argument(
        "--y_values",
        type=str,
        default=format_default_value_list(SCRIPT_SWEEP_DEFAULTS["y_values"]),
        help="Comma-separated values for the y-axis parameter.",
    )
    parser.add_argument("--nx", type=int, default=None, help="Optional x grid size if using x_min/x_max.")
    parser.add_argument("--ny", type=int, default=None, help="Optional y grid size if using y_min/y_max.")
    parser.add_argument("--x_min", type=float, default=None, help="Optional x-axis minimum if generating a numeric grid.")
    parser.add_argument("--x_max", type=float, default=None, help="Optional x-axis maximum if generating a numeric grid.")
    parser.add_argument("--y_min", type=float, default=None, help="Optional y-axis minimum if generating a numeric grid.")
    parser.add_argument("--y_max", type=float, default=None, help="Optional y-axis maximum if generating a numeric grid.")
    parser.add_argument(
        "--tau_max",
        type=float,
        default=SCRIPT_SWEEP_DEFAULTS["tau_max"],
        help="Final ODE time. Set the script default to `None` to use the nominal training horizon n_epochs * learning_rate * batches_per_epoch.",
    )
    parser.add_argument("--out", type=str, default=SCRIPT_SWEEP_DEFAULTS["out"])
    parser.add_argument("--out_data", type=str, default=SCRIPT_SWEEP_DEFAULTS["out_data"], help="Optional `.npz` path for final theory metrics and phases.")
    parser.add_argument("--loss_small_thresh", type=float, default=SCRIPT_CLASSIFIER_THRESHOLDS["loss_small_thresh"])
    parser.add_argument("--small_thresh", type=float, default=SCRIPT_CLASSIFIER_THRESHOLDS["small_thresh"])
    parser.add_argument("--active_thresh", type=float, default=SCRIPT_CLASSIFIER_THRESHOLDS["active_thresh"])
    parser.add_argument("--rho_thresh", type=float, default=SCRIPT_CLASSIFIER_THRESHOLDS["rho_thresh"])

    for field in fields(ExperimentConfig):
        default = getattr(SCRIPT_EXPERIMENT_DEFAULTS, field.name)
        parser.add_argument(f"--{field.name}", type=type(default), default=default)

    return parser


def build_configs_from_args(args: argparse.Namespace) -> Tuple[ExperimentConfig, str, str, Tuple[Any, ...], Tuple[Any, ...]]:
    x_values = parse_value_list(args.x_values, args.vary_x) if args.x_values else maybe_build_linspace(args.vary_x, args.nx, args.x_min, args.x_max)
    y_values = parse_value_list(args.y_values, args.vary_y) if args.y_values else maybe_build_linspace(args.vary_y, args.ny, args.y_min, args.y_max)
    if x_values is None or y_values is None:
        raise ValueError("Provide either explicit value lists (`--x_values`, `--y_values`) or numeric grid ranges (`--nx`, `--x_min`, `--x_max`, ...).")
    if args.vary_x == args.vary_y:
        raise ValueError("`--vary_x` and `--vary_y` must be different config fields.")

    base_kwargs = {field.name: getattr(args, field.name) for field in fields(ExperimentConfig)}
    base_config = ExperimentConfig(**base_kwargs)
    return base_config, args.vary_x, args.vary_y, tuple(x_values), tuple(y_values)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def nominal_tau_max(config: ExperimentConfig) -> float:
    n_samples = max(int(config.n_samples), 1)
    batch_size = min(max(int(config.batch_size), 1), n_samples)
    epoch_size = int(config.epoch_size) if int(config.epoch_size) > 0 else n_samples
    batches_per_epoch = max(int(np.ceil(epoch_size / batch_size)), 1)
    return float(config.n_epochs) * float(config.learning_rate) * float(batches_per_epoch)


def cell_edges(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Axis values must be a non-empty 1D sequence.")
    if arr.size == 1:
        return np.asarray([arr[0] - 0.5, arr[0] + 0.5], dtype=float)

    mids = 0.5 * (arr[:-1] + arr[1:])
    left = arr[0] - 0.5 * (arr[1] - arr[0])
    right = arr[-1] + 0.5 * (arr[-1] - arr[-2])
    return np.concatenate(([left], mids, [right]))


def compute_initial_state(config: ExperimentConfig, device: torch.device) -> Tuple[Dict[str, Any], np.ndarray]:
    np.random.seed(int(config.seed))
    torch.manual_seed(int(config.seed))

    params = build_shared_params(config)
    X, y, U, v, lam = generate_dataset(config, params, N_DIM, R_DIM)
    ae, lat_dis = make_linear_fader(device)
    initial_obs, _ = compute_measured_observables(
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
    return params, observables_to_state(initial_obs)


def run_single_grid_point(
    config: ExperimentConfig,
    tau_max: float,
    device: torch.device,
) -> Tuple[Dict[str, float], str]:
    params, x0 = compute_initial_state(config, device)

    if tau_max <= 0.0:
        final_metrics = scalarize_state(x0, params)
    else:
        t_eval = np.linspace(0.0, tau_max, max(int(config.theory_points), 2))
        theory_states = integrate_theory(x0, params, t_eval)
        final_metrics = scalarize_state(theory_states[-1], params)

    phase = classify_phase(final_metrics, source="theory")
    return final_metrics, phase


def plot_ode_phase_diagram(
    x_values: Sequence[float],
    y_values: Sequence[float],
    phase_grid: np.ndarray,
    vary_x: str,
    vary_y: str,
    out_file: str,
) -> None:
    phases = ordered_phases(phase_grid)
    colors = [DEFAULT_PHASE_COLORS.get(phase, "#95a5a6") for phase in phases]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(len(phases) + 1) - 0.5, cmap.N)

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    x_edges = cell_edges(x)
    y_edges = cell_edges(y)
    grid = phase_to_int(phase_grid, phases).T

    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        interpolation="nearest",
    )
    ax.set_title("ODE Phase Diagram", color="white", fontsize=12)
    ax.set_xlabel(vary_x, color="white")
    ax.set_ylabel(vary_y, color="white")
    ax.grid(True, color=GRID, alpha=0.45)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    patches = [mpatches.Patch(color=colors[idx], label=phase) for idx, phase in enumerate(phases)]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=min(len(phases), 4),
        facecolor=PANEL,
        edgecolor="#555",
        labelcolor="white",
        framealpha=0.92,
    )
    fig.suptitle(f"ODE Phase Diagram: {vary_x} x {vary_y}", color="white", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_file, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def save_grid_data(
    out_data: str,
    vary_x: str,
    vary_y: str,
    x_values: Tuple[Any, ...],
    y_values: Tuple[Any, ...],
    phase_grid: np.ndarray,
    metric_grids: Dict[str, np.ndarray],
    tau_grid: np.ndarray,
) -> None:
    ensure_parent_dir(out_data)
    save_dict: Dict[str, Any] = {
        "vary_x": np.asarray(vary_x),
        "vary_y": np.asarray(vary_y),
        "x_values": np.asarray(x_values),
        "y_values": np.asarray(y_values),
        "final_tau": tau_grid,
        "metric_names": np.asarray(FINAL_METRIC_NAMES, dtype=str),
        "theory_phase": phase_grid.astype(str),
    }
    for metric_name, values in metric_grids.items():
        save_dict[f"theory_{metric_name}"] = values
    np.savez(out_data, **save_dict)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    CLASSIFIER_THRESHOLDS["loss_small"] = float(args.loss_small_thresh)
    CLASSIFIER_THRESHOLDS["small"] = float(args.small_thresh)
    CLASSIFIER_THRESHOLDS["active"] = float(args.active_thresh)
    CLASSIFIER_THRESHOLDS["rho"] = float(args.rho_thresh)

    base_config, vary_x, vary_y, x_values, y_values = build_configs_from_args(args)
    tau_max = float(args.tau_max) if args.tau_max is not None else nominal_tau_max(base_config)

    if base_config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    nx = len(x_values)
    ny = len(y_values)
    phase_grid = np.empty((nx, ny), dtype=object)
    metric_grids = {
        name: np.full((nx, ny), np.nan, dtype=float)
        for name in FINAL_METRIC_NAMES
    }
    tau_grid = np.full((nx, ny), tau_max, dtype=float)

    total = nx * ny
    counter = 0
    for ix, x_value in enumerate(x_values):
        for iy, y_value in enumerate(y_values):
            counter += 1
            config = replace(base_config, **{vary_x: x_value, vary_y: y_value})
            print(f"[{counter}/{total}] {vary_x}={x_value}, {vary_y}={y_value}, tau_max={tau_max:.6g}")
            final_metrics, final_phase = run_single_grid_point(config, tau_max, device)
            phase_grid[ix, iy] = final_phase
            for metric_name in FINAL_METRIC_NAMES:
                metric_grids[metric_name][ix, iy] = float(final_metrics[metric_name])
            print(
                f"  phase={final_phase} "
                f"rec={final_metrics['reconstruction_error']:.6g} "
                f"rho={final_metrics['rho']:.6g}"
            )

    ensure_parent_dir(args.out)
    plot_ode_phase_diagram(x_values, y_values, phase_grid, vary_x, vary_y, args.out)
    print(f"Saved -> {args.out}")

    if args.out_data:
        save_grid_data(args.out_data, vary_x, vary_y, x_values, y_values, phase_grid, metric_grids, tau_grid)
        print(f"Saved -> {args.out_data}")


if __name__ == "__main__":
    main()
