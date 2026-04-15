#!/usr/bin/env python3
"""Shared phase classification utilities for Branch-B sweeps."""

from __future__ import annotations

from typing import Dict, List, Mapping

import numpy as np


EPS = 1e-12

DEFAULT_PHASES = (
    "no_learning",
    "label_loss",
    "disentangled_learning",
    "entangled_learning",
    "unclassified",
)

DEFAULT_PHASE_COLORS = {
    "no_learning": "#7f8c8d",
    "label_loss": "#e67e22",
    "disentangled_learning": "#2ecc71",
    "entangled_learning": "#3498db",
    "unclassified": "#34495e",
}

CLASSIFIER_CONFIG = {
    "signal_floor_abs": 0.15,
    "signal_floor_rel": 0.25,
    "label_floor_abs": 0.02,
    "label_floor_rel": 0.25,
    "dominance_high": 0.67,
}


def _safe_float(value: object) -> float:
    return float(value) if np.isfinite(value) else np.nan


def harmonic_mean_nonnegative(x: float, y: float) -> float:
    x = max(float(x), 0.0)
    y = max(float(y), 0.0)
    if x <= 0.0 or y <= 0.0:
        return 0.0
    return 2.0 * x * y / max(x + y, EPS)


def derive_phase_metrics(cell_metrics: Mapping[str, float]) -> Dict[str, float]:
    rho = abs(_safe_float(cell_metrics.get("rho", np.nan)))
    latent_label_coupling = abs(
        _safe_float(
            cell_metrics.get(
                "latent_label_coupling",
                _safe_float(cell_metrics.get("norm_s", np.nan)) * _safe_float(cell_metrics.get("norm_a", np.nan)),
            )
        )
    )
    m_tilde = _safe_float(cell_metrics.get("M_tilde", np.nan))
    n_tilde = _safe_float(cell_metrics.get("N_tilde", np.nan))
    signal_score = harmonic_mean_nonnegative(m_tilde, n_tilde)
    label_score = rho + latent_label_coupling
    explicit_fraction = rho / max(label_score, EPS)
    entangled_fraction = latent_label_coupling / max(label_score, EPS)
    b_norm = _safe_float(cell_metrics.get("b_norm", np.sqrt(max(_safe_float(cell_metrics.get("m", 0.0)), 0.0))))
    b_perp_norm = _safe_float(
        cell_metrics.get(
            "b_perp_norm",
            np.sqrt(max(_safe_float(cell_metrics.get("m", 0.0)) - _safe_float(cell_metrics.get("rho", 0.0)) ** 2, 0.0)),
        )
    )
    return {
        "signal_score": signal_score,
        "label_score": label_score,
        "explicit_label_score": rho,
        "latent_label_score": latent_label_coupling,
        "explicit_fraction": explicit_fraction,
        "entangled_fraction": entangled_fraction,
        "b_norm": b_norm,
        "b_perp_norm": b_perp_norm,
    }


def fit_classifier_thresholds(metric_grids: Mapping[str, np.ndarray]) -> Dict[str, float]:
    signal_scores: List[float] = []
    label_scores: List[float] = []
    sample = next(iter(metric_grids.values()))
    nx, ny = sample.shape
    for ix in range(nx):
        for iy in range(ny):
            cell_metrics = {name: float(values[ix, iy]) for name, values in metric_grids.items()}
            derived = derive_phase_metrics(cell_metrics)
            signal_scores.append(derived["signal_score"])
            label_scores.append(derived["label_score"])

    signal_arr = np.asarray(signal_scores, dtype=float)
    label_arr = np.asarray(label_scores, dtype=float)
    signal_scale = float(np.nanpercentile(signal_arr, 90)) if np.isfinite(signal_arr).any() else 0.0
    label_scale = float(np.nanpercentile(label_arr, 90)) if np.isfinite(label_arr).any() else 0.0
    return {
        "signal_floor": max(CLASSIFIER_CONFIG["signal_floor_abs"], CLASSIFIER_CONFIG["signal_floor_rel"] * signal_scale),
        "label_floor": max(CLASSIFIER_CONFIG["label_floor_abs"], CLASSIFIER_CONFIG["label_floor_rel"] * label_scale),
        "dominance_high": float(CLASSIFIER_CONFIG["dominance_high"]),
    }


def classify_phase(cell_metrics: Mapping[str, float], source: str, thresholds: Mapping[str, float]) -> str:
    _ = source
    derived = derive_phase_metrics(cell_metrics)
    if not all(np.isfinite(value) for value in derived.values()):
        return "unclassified"

    signal_active = derived["signal_score"] >= float(thresholds["signal_floor"])
    label_active = derived["label_score"] >= float(thresholds["label_floor"])

    if not signal_active:
        return "no_learning"
    if not label_active:
        return "label_loss"
    if derived["explicit_fraction"] >= float(thresholds["dominance_high"]):
        return "disentangled_learning"
    return "entangled_learning"


def build_phase_grid(metric_grids: Mapping[str, np.ndarray], source: str) -> np.ndarray:
    sample = next(iter(metric_grids.values()))
    nx, ny = sample.shape
    thresholds = fit_classifier_thresholds(metric_grids)
    phase_grid = np.empty((nx, ny), dtype=object)
    for ix in range(nx):
        for iy in range(ny):
            cell_metrics = {name: float(values[ix, iy]) for name, values in metric_grids.items()}
            phase_grid[ix, iy] = classify_phase(cell_metrics, source=source, thresholds=thresholds)
    return phase_grid


def build_derived_metric_grids(metric_grids: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    sample = next(iter(metric_grids.values()))
    nx, ny = sample.shape
    derived_names = (
        "signal_score",
        "label_score",
        "explicit_label_score",
        "latent_label_score",
        "explicit_fraction",
        "entangled_fraction",
        "b_norm",
        "b_perp_norm",
    )
    derived_grids = {name: np.full((nx, ny), np.nan, dtype=float) for name in derived_names}
    for ix in range(nx):
        for iy in range(ny):
            cell_metrics = {name: float(values[ix, iy]) for name, values in metric_grids.items()}
            derived = derive_phase_metrics(cell_metrics)
            for name in derived_names:
                derived_grids[name][ix, iy] = derived[name]
    return derived_grids
