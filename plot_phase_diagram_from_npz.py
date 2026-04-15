#!/usr/bin/env python3
"""
Plot Fader and ODE phase diagrams from a saved grid `.npz`.

Phase classification is delegated to `phase_classifier.py` so the ODE-only and
measured-vs-theory plotting paths use the same adaptive rules.
"""

import argparse
import os
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from phase_classifier import (
    CLASSIFIER_CONFIG,
    DEFAULT_PHASES,
    DEFAULT_PHASE_COLORS,
    build_derived_metric_grids,
    build_phase_grid,
)

BG = "#1a1a2e"
PANEL = "#0f0f23"
GRID = "#2a2a4a"


def load_grid(npz_path: str) -> np.lib.npyio.NpzFile:
    return np.load(npz_path, allow_pickle=False)


def decode_scalar(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if np.isscalar(value):
        return str(value)
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return decode_scalar(value.item())
    return str(value)


def final_metric_grid(data: np.lib.npyio.NpzFile, prefix: str, metric: str, counts_key: str) -> np.ndarray:
    series = np.asarray(data[f"{prefix}_{metric}"], dtype=float)
    counts = np.asarray(data[counts_key], dtype=int)
    if series.ndim != 3:
        raise ValueError(f"Expected 3D series for {prefix}_{metric}, got shape {series.shape}")

    nx, ny, _ = series.shape
    out = np.full((nx, ny), np.nan, dtype=float)
    for ix in range(nx):
        for iy in range(ny):
            count = int(counts[ix, iy])
            if count <= 0:
                continue
            out[ix, iy] = series[ix, iy, count - 1]
    return out


def build_final_metric_dicts(data: np.lib.npyio.NpzFile) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    metric_names = [decode_scalar(x) for x in np.asarray(data["metric_names"])]
    measured = {
        metric: final_metric_grid(data, "measured", metric, "measure_counts")
        for metric in metric_names
    }
    theory = {
        metric: final_metric_grid(data, "theory", metric, "measure_counts")
        for metric in metric_names
    }
    return measured, theory


def ordered_phases(*phase_grids: np.ndarray) -> List[str]:
    seen = []
    for phase in DEFAULT_PHASES:
        if phase not in seen:
            seen.append(phase)
    for grid in phase_grids:
        for phase in grid.ravel():
            phase_str = str(phase)
            if phase_str not in seen:
                seen.append(phase_str)
    return seen


def phase_to_int(phase_grid: np.ndarray, phases: List[str]) -> np.ndarray:
    mapping = {phase: idx for idx, phase in enumerate(phases)}
    out = np.zeros(phase_grid.shape, dtype=int)
    for phase, idx in mapping.items():
        out[phase_grid == phase] = idx
    return out


def plot_phase_diagrams(
    x_values: np.ndarray,
    y_values: np.ndarray,
    fader_grid: np.ndarray,
    theory_grid: np.ndarray,
    vary_x: str,
    vary_y: str,
    out_file: str,
) -> None:
    phases = ordered_phases(fader_grid, theory_grid)
    colors = [DEFAULT_PHASE_COLORS.get(phase, "#95a5a6") for phase in phases]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(len(phases) + 1) - 0.5, cmap.N)

    x = np.asarray(x_values)
    y = np.asarray(y_values)
    fader_int = phase_to_int(fader_grid, phases).T
    theory_int = phase_to_int(theory_grid, phases).T

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    fig.patch.set_facecolor(BG)

    for ax, grid, title in (
        (axes[0], fader_int, "Fader Phase Diagram"),
        (axes[1], theory_int, "ODE Phase Diagram"),
    ):
        ax.set_facecolor(PANEL)
        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=[x[0], x[-1], y[0], y[-1]],
            interpolation="nearest",
        )
        ax.set_title(title, color="white", fontsize=12)
        ax.set_xlabel(vary_x, color="white")
        ax.grid(True, color=GRID, alpha=0.45)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    axes[0].set_ylabel(vary_y, color="white")

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
    fig.suptitle(f"Phase Diagram Comparison: {vary_x} x {vary_y}", color="white", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_file, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_reference_heatmaps(
    x_values: np.ndarray,
    y_values: np.ndarray,
    metric_grids: Dict[str, np.ndarray],
    vary_x: str,
    vary_y: str,
    out_file: str,
) -> None:
    metrics = [
        ("signal_score", "signal score"),
        ("latent_label_score", "|a.s|"),
        ("explicit_label_score", "|rho|"),
        ("explicit_fraction", "explicit fraction"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)
    fig.patch.set_facecolor(BG)

    for row_idx, (source, title_prefix) in enumerate((("measured", "Fader"), ("theory", "ODE"))):
        for col_idx, (metric_key, metric_title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            grid = metric_grids[f"{source}:{metric_key}"].T
            im = ax.imshow(
                grid,
                origin="lower",
                aspect="auto",
                extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]],
                interpolation="nearest",
                cmap="viridis",
            )
            ax.set_facecolor(PANEL)
            ax.set_title(f"{title_prefix} {metric_title}", color="white", fontsize=11)
            ax.tick_params(colors="white")
            ax.grid(True, color=GRID, alpha=0.35)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")
            if row_idx == 1:
                ax.set_xlabel(vary_x, color="white")
            if col_idx == 0:
                ax.set_ylabel(vary_y, color="white")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
            cb.outline.set_edgecolor("#444")

    fig.suptitle("Final Metric Heatmaps", color="white", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_file, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Fader and ODE phase diagrams from a grid `.npz`.")
    parser.add_argument("--input", type=str, required=True, help="Path to the grid `.npz` generated by fader_phase_diagrams_repo_linear.py")
    parser.add_argument("--out_phase", type=str, default=None, help="Output PNG for the phase-diagram comparison")
    parser.add_argument("--out_metrics", type=str, default=None, help="Optional output PNG for reference metric heatmaps")
    parser.add_argument("--signal_floor_abs", type=float, default=CLASSIFIER_CONFIG["signal_floor_abs"])
    parser.add_argument("--signal_floor_rel", type=float, default=CLASSIFIER_CONFIG["signal_floor_rel"])
    parser.add_argument("--label_floor_abs", type=float, default=CLASSIFIER_CONFIG["label_floor_abs"])
    parser.add_argument("--label_floor_rel", type=float, default=CLASSIFIER_CONFIG["label_floor_rel"])
    parser.add_argument("--dominance_high", type=float, default=CLASSIFIER_CONFIG["dominance_high"])
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    CLASSIFIER_CONFIG["signal_floor_abs"] = float(args.signal_floor_abs)
    CLASSIFIER_CONFIG["signal_floor_rel"] = float(args.signal_floor_rel)
    CLASSIFIER_CONFIG["label_floor_abs"] = float(args.label_floor_abs)
    CLASSIFIER_CONFIG["label_floor_rel"] = float(args.label_floor_rel)
    CLASSIFIER_CONFIG["dominance_high"] = float(args.dominance_high)

    data = load_grid(args.input)
    x_values = np.asarray(data["x_values"], dtype=float)
    y_values = np.asarray(data["y_values"], dtype=float)
    vary_x = decode_scalar(data["vary_x"])
    vary_y = decode_scalar(data["vary_y"])

    measured_metrics, theory_metrics = build_final_metric_dicts(data)
    fader_grid = build_phase_grid(measured_metrics, source="fader")
    theory_grid = build_phase_grid(theory_metrics, source="theory")

    base, _ = os.path.splitext(args.input)
    out_phase = args.out_phase or f"{base}_phase_diagrams.png"
    out_metrics = args.out_metrics or f"{base}_final_metrics.png"

    plot_phase_diagrams(x_values, y_values, fader_grid, theory_grid, vary_x, vary_y, out_phase)

    merged_metric_grids = {}
    for name, values in measured_metrics.items():
        merged_metric_grids[f"measured:{name}"] = values
    for name, values in theory_metrics.items():
        merged_metric_grids[f"theory:{name}"] = values
    for name, values in build_derived_metric_grids(measured_metrics).items():
        merged_metric_grids[f"measured:{name}"] = values
    for name, values in build_derived_metric_grids(theory_metrics).items():
        merged_metric_grids[f"theory:{name}"] = values
    plot_reference_heatmaps(x_values, y_values, merged_metric_grids, vary_x, vary_y, out_metrics)

    print(f"Saved -> {out_phase}")
    print(f"Saved -> {out_metrics}")


if __name__ == "__main__":
    main()
