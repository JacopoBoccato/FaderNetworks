#!/usr/bin/env python3
"""
Run an n x m grid of strict-linear Fader / Branch-B comparisons.

For each grid point, this script:
1. Builds a synthetic Gaussian teacher dataset.
2. Trains the microscopic linear Fader with minibatch saddle updates.
3. Stops early when the full-dataset reconstruction error is stable.
4. Integrates the matching ODE up to the same final time.
5. Saves all measured / theoretical plot metrics and experiment metadata to
   a single `.npz` file.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from phase_diagram import (
    OUT_DATA,
    D_DIM,
    N_DIM,
    R_DIM,
    ExperimentConfig as SingleRunConfig,
    build_shared_params,
    classifier_strength,
    compute_measured_observables,
    generate_dataset,
    integrate_theory,
    make_linear_fader,
    microscopic_objectives,
    microscopic_saddle_step,
    observables_to_state,
    scalarize_observables,
    scalarize_state,
)


PLOT_METRICS: Tuple[str, ...] = (
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
    "b_norm",
    "b_perp_norm",
    "latent_label_coupling",
    "reconstruction_error",
    "M_tilde",
    "N_tilde",
)

CONFIG_FIELDS = {field.name: field.type for field in fields(SingleRunConfig)}
DEFAULT_SINGLE_RUN = SingleRunConfig()


@dataclass(frozen=True)
class SweepConfig:
    vary_x: str
    vary_y: str
    x_values: Tuple[Any, ...]
    y_values: Tuple[Any, ...]
    max_epochs: int
    conv_window: int
    conv_min_epochs: int
    conv_rel_improve_tol: float
    conv_stability_tol: float
    conv_check_every: int
    workers: int
    out_data: str


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
    parser = argparse.ArgumentParser(description="Run a grid of strict-linear Fader / Branch-B comparisons.")

    parser.add_argument("--vary_x", type=str, required=True, choices=sorted(CONFIG_FIELDS))
    parser.add_argument("--vary_y", type=str, required=True, choices=sorted(CONFIG_FIELDS))
    parser.add_argument("--x_values", type=str, default=None, help="Comma-separated values for the x-axis parameter.")
    parser.add_argument("--y_values", type=str, default=None, help="Comma-separated values for the y-axis parameter.")
    parser.add_argument("--nx", type=int, default=None, help="Optional x grid size if using x_min/x_max.")
    parser.add_argument("--ny", type=int, default=None, help="Optional y grid size if using y_min/y_max.")
    parser.add_argument("--x_min", type=float, default=None, help="Optional x-axis minimum if generating a numeric grid.")
    parser.add_argument("--x_max", type=float, default=None, help="Optional x-axis maximum if generating a numeric grid.")
    parser.add_argument("--y_min", type=float, default=None, help="Optional y-axis minimum if generating a numeric grid.")
    parser.add_argument("--y_max", type=float, default=None, help="Optional y-axis maximum if generating a numeric grid.")
    parser.add_argument("--out_data", type=str, default=os.path.join("results", "branch_b_grid_results.npz"))
    parser.add_argument("--workers", type=int, default=1, help="Accepted for compatibility; the sweep currently runs sequentially.")
    parser.add_argument("--max_epochs", type=int, default=None, help="Upper bound on training epochs per grid point.")
    parser.add_argument("--conv_window", type=int, default=10)
    parser.add_argument("--conv_min_epochs", type=int, default=40)
    parser.add_argument("--conv_rel_improve_tol", type=float, default=0.02)
    parser.add_argument("--conv_stability_tol", type=float, default=0.05)
    parser.add_argument("--conv_check_every", type=int, default=1)

    base = DEFAULT_SINGLE_RUN
    for field in fields(SingleRunConfig):
        default = getattr(base, field.name)
        parser.add_argument(f"--{field.name}", type=type(default), default=default)

    return parser


def build_configs_from_args(args: argparse.Namespace) -> Tuple[SingleRunConfig, SweepConfig]:
    x_values = parse_value_list(args.x_values, args.vary_x) if args.x_values else maybe_build_linspace(args.vary_x, args.nx, args.x_min, args.x_max)
    y_values = parse_value_list(args.y_values, args.vary_y) if args.y_values else maybe_build_linspace(args.vary_y, args.ny, args.y_min, args.y_max)
    if x_values is None or y_values is None:
        raise ValueError("Provide either explicit value lists (`--x_values`, `--y_values`) or numeric grid ranges (`--nx`, `--x_min`, `--x_max`, ...).")

    base_kwargs = {field.name: getattr(args, field.name) for field in fields(SingleRunConfig)}
    base_config = SingleRunConfig(**base_kwargs)

    sweep_config = SweepConfig(
        vary_x=args.vary_x,
        vary_y=args.vary_y,
        x_values=tuple(x_values),
        y_values=tuple(y_values),
        max_epochs=int(args.max_epochs if args.max_epochs is not None else base_config.n_epochs),
        conv_window=max(int(args.conv_window), 2),
        conv_min_epochs=max(int(args.conv_min_epochs), 1),
        conv_rel_improve_tol=float(args.conv_rel_improve_tol),
        conv_stability_tol=float(args.conv_stability_tol),
        conv_check_every=max(int(args.conv_check_every), 1),
        workers=max(int(args.workers), 1),
        out_data=args.out_data,
    )

    return base_config, sweep_config


def convergence_stats(loss_history: Sequence[float], window: int) -> Optional[Tuple[float, float]]:
    if len(loss_history) < 2 * window:
        return None
    prev = np.asarray(loss_history[-2 * window:-window], dtype=float)
    last = np.asarray(loss_history[-window:], dtype=float)
    prev_mean = float(np.mean(prev))
    last_mean = float(np.mean(last))
    rel_improve = abs(prev_mean - last_mean) / max(abs(prev_mean), 1e-12)
    stability = float(np.std(last) / max(abs(last_mean), 1e-12))
    return rel_improve, stability


def train_until_stable(
    X: torch.Tensor,
    y: torch.Tensor,
    U: np.ndarray,
    v: np.ndarray,
    lam: np.ndarray,
    params: Dict[str, Any],
    config: SingleRunConfig,
    sweep: SweepConfig,
    device: torch.device,
) -> Dict[str, Any]:
    ae, lat_dis = make_linear_fader(device)

    X_dev = X.to(device)
    y_dev = y.to(device)
    n_samples = X_dev.size(0)
    batch_size = min(max(int(config.batch_size), 1), n_samples)
    epoch_size = int(config.epoch_size) if int(config.epoch_size) > 0 else int(n_samples)
    n_batches_per_epoch = max(int(np.ceil(epoch_size / batch_size)), 1)
    measure_every_steps = max(int(config.measure_every_batches), 1)
    joint_update_dt = float(config.learning_rate)

    measured = []
    times = [0.0]
    sgd_times = [0.0]
    epoch_ae_losses: List[float] = []
    epoch_clf_losses: List[float] = []
    epoch_rec_losses: List[float] = []
    gamma_history: List[float] = []

    initial_obs, initial_rec = compute_measured_observables(
        ae, lat_dis, X, y, U, v, lam, params["eta"], params["g"], device
    )
    measured.append(scalarize_observables(initial_obs, initial_rec))
    x0 = observables_to_state(initial_obs)

    elapsed_time = 0.0
    step_idx = 0
    stable_reached = False
    stable_epoch = sweep.max_epochs
    final_rel_improve = np.nan
    final_stability = np.nan

    for epoch in range(1, sweep.max_epochs + 1):
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

            microscopic_saddle_step(ae, lat_dis, batch_x, batch_y, elapsed_time, params, device)
            step_idx += 1
            elapsed_time += joint_update_dt

            if step_idx % measure_every_steps == 0:
                obs, rec = compute_measured_observables(
                    ae, lat_dis, X, y, U, v, lam, params["eta"], params["g"], device
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

        if (
            epoch >= sweep.conv_min_epochs
            and epoch % sweep.conv_check_every == 0
        ):
            stats = convergence_stats(epoch_rec_losses, sweep.conv_window)
            if stats is not None:
                final_rel_improve, final_stability = stats
                if (
                    final_rel_improve <= sweep.conv_rel_improve_tol
                    and final_stability <= sweep.conv_stability_tol
                ):
                    stable_reached = True
                    stable_epoch = epoch
                    break

    if times[-1] < elapsed_time:
        obs, rec = compute_measured_observables(
            ae, lat_dis, X, y, U, v, lam, params["eta"], params["g"], device
        )
        measured.append(scalarize_observables(obs, rec))
        times.append(elapsed_time)
        sgd_times.append(elapsed_time)

    return {
        "times": np.asarray(times, dtype=float),
        "times_sgd": np.asarray(sgd_times, dtype=float),
        "loss_history": {
            "ae_loss": np.asarray(epoch_ae_losses, dtype=float),
            "clf_loss": np.asarray(epoch_clf_losses, dtype=float),
            "rec_loss": np.asarray(epoch_rec_losses, dtype=float),
            "gamma": np.asarray(gamma_history, dtype=float),
        },
        "measured": measured,
        "x0": x0,
        "matched_dt": float(joint_update_dt),
        "stable_reached": bool(stable_reached),
        "stable_epoch": int(stable_epoch),
        "used_epochs": int(len(epoch_rec_losses)),
        "used_steps": int(step_idx),
        "final_tau": float(elapsed_time),
        "final_rel_improve": float(final_rel_improve),
        "final_stability": float(final_stability),
    }


def run_single_grid_point(
    config: SingleRunConfig,
    sweep: SweepConfig,
    device: torch.device,
) -> Dict[str, Any]:
    params = build_shared_params(config)
    X, y, U, v, lam = generate_dataset(config, params, N_DIM, R_DIM)

    train_result = train_until_stable(X, y, U, v, lam, params, config, sweep, device)

    theory_times_dense = np.linspace(0.0, max(train_result["final_tau"], 1e-12), int(config.theory_points))
    theory_states_dense = integrate_theory(train_result["x0"], params, theory_times_dense)
    theory_list_dense = [scalarize_state(x, params) for x in theory_states_dense]

    keys = list(train_result["measured"][0].keys())
    measured_hist = {
        key: np.asarray([row[key] for row in train_result["measured"]], dtype=float)
        for key in keys
    }
    theory_hist_dense = {
        key: np.asarray([row[key] for row in theory_list_dense], dtype=float)
        for key in keys
    }
    theory_hist_on_measured = {
        key: np.interp(train_result["times"], theory_times_dense, theory_hist_dense[key])
        for key in keys
    }

    return {
        "config": config,
        "params": params,
        "train": train_result,
        "measured_hist": measured_hist,
        "theory_hist_dense": theory_hist_dense,
        "theory_hist": theory_hist_on_measured,
        "theory_times_dense": theory_times_dense,
    }


def pad_1d_series(series_list: List[np.ndarray], fill_value: float = np.nan) -> Tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([len(x) for x in series_list], dtype=int)
    max_len = int(lengths.max(initial=0))
    out = np.full((len(series_list), max_len), fill_value, dtype=float)
    for idx, arr in enumerate(series_list):
        out[idx, : len(arr)] = arr
    return out, lengths


def pad_metric_cube(metric_list: List[Dict[str, np.ndarray]], metric_names: Sequence[str], fill_value: float = np.nan) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    lengths = np.asarray([len(next(iter(item.values()))) for item in metric_list], dtype=int)
    max_len = int(lengths.max(initial=0))
    out = {
        name: np.full((len(metric_list), max_len), fill_value, dtype=float)
        for name in metric_names
    }
    for idx, metric_dict in enumerate(metric_list):
        for name in metric_names:
            arr = np.asarray(metric_dict[name], dtype=float)
            out[name][idx, : len(arr)] = arr
    return out, lengths


def reshape_grid(flat: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return flat.reshape(nx, ny, *flat.shape[1:])


def reshape_metric_grid(metric_dict: Dict[str, np.ndarray], nx: int, ny: int) -> Dict[str, np.ndarray]:
    return {
        name: values.reshape(nx, ny, *values.shape[1:])
        for name, values in metric_dict.items()
    }


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    base_config, sweep = build_configs_from_args(args)

    np.random.seed(base_config.seed)
    torch.manual_seed(base_config.seed)

    if base_config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    flat_results: List[Dict[str, Any]] = []
    nx = len(sweep.x_values)
    ny = len(sweep.y_values)

    if sweep.workers != 1:
        print(f"workers={sweep.workers} requested, but the sweep currently runs sequentially.")

    total = nx * ny
    counter = 0
    for ix, x_value in enumerate(sweep.x_values):
        for iy, y_value in enumerate(sweep.y_values):
            counter += 1
            config = replace(base_config, **{
                sweep.vary_x: x_value,
                sweep.vary_y: y_value,
                "n_epochs": sweep.max_epochs,
            })
            print(
                f"[{counter}/{total}] {sweep.vary_x}={x_value}, {sweep.vary_y}={y_value} "
                f"(max_epochs={sweep.max_epochs})"
            )
            result = run_single_grid_point(config, sweep, device)
            result["grid_index"] = (ix, iy)
            flat_results.append(result)
            print(
                f"  epochs={result['train']['used_epochs']} stable={result['train']['stable_reached']} "
                f"tau={result['train']['final_tau']:.6g} "
                f"final_rec={result['train']['loss_history']['rec_loss'][-1]:.6g}"
            )

    times_flat, measure_counts = pad_1d_series([item["train"]["times"] for item in flat_results])
    times_sgd_flat, _ = pad_1d_series([item["train"]["times_sgd"] for item in flat_results])
    theory_times_flat, theory_counts = pad_1d_series([item["theory_times_dense"] for item in flat_results])

    measured_metric_flat, _ = pad_metric_cube([item["measured_hist"] for item in flat_results], PLOT_METRICS)
    theory_metric_flat, _ = pad_metric_cube([item["theory_hist"] for item in flat_results], PLOT_METRICS)
    theory_dense_metric_flat, _ = pad_metric_cube([item["theory_hist_dense"] for item in flat_results], PLOT_METRICS)

    ae_loss_flat, epoch_counts = pad_1d_series([item["train"]["loss_history"]["ae_loss"] for item in flat_results])
    clf_loss_flat, _ = pad_1d_series([item["train"]["loss_history"]["clf_loss"] for item in flat_results])
    rec_loss_flat, _ = pad_1d_series([item["train"]["loss_history"]["rec_loss"] for item in flat_results])
    gamma_flat, _ = pad_1d_series([item["train"]["loss_history"]["gamma"] for item in flat_results])

    stable_reached = np.asarray([item["train"]["stable_reached"] for item in flat_results], dtype=bool).reshape(nx, ny)
    stable_epoch = np.asarray([item["train"]["stable_epoch"] for item in flat_results], dtype=int).reshape(nx, ny)
    used_epochs = np.asarray([item["train"]["used_epochs"] for item in flat_results], dtype=int).reshape(nx, ny)
    used_steps = np.asarray([item["train"]["used_steps"] for item in flat_results], dtype=int).reshape(nx, ny)
    final_tau = np.asarray([item["train"]["final_tau"] for item in flat_results], dtype=float).reshape(nx, ny)
    final_rel_improve = np.asarray([item["train"]["final_rel_improve"] for item in flat_results], dtype=float).reshape(nx, ny)
    final_stability = np.asarray([item["train"]["final_stability"] for item in flat_results], dtype=float).reshape(nx, ny)
    matched_dt = np.asarray([item["train"]["matched_dt"] for item in flat_results], dtype=float).reshape(nx, ny)

    config_json = json.dumps(asdict(base_config), sort_keys=True)
    sweep_json = json.dumps({
        "vary_x": sweep.vary_x,
        "vary_y": sweep.vary_y,
        "x_values": list(sweep.x_values),
        "y_values": list(sweep.y_values),
        "max_epochs": sweep.max_epochs,
        "conv_window": sweep.conv_window,
        "conv_min_epochs": sweep.conv_min_epochs,
        "conv_rel_improve_tol": sweep.conv_rel_improve_tol,
        "conv_stability_tol": sweep.conv_stability_tol,
        "conv_check_every": sweep.conv_check_every,
        "workers": sweep.workers,
    }, sort_keys=True)

    config_grids: Dict[str, np.ndarray] = {}
    for field in fields(SingleRunConfig):
        values = np.asarray([getattr(item["config"], field.name) for item in flat_results])
        config_grids[field.name] = values.reshape(nx, ny)

    shared_param_grids: Dict[str, np.ndarray] = {}
    scalar_param_names = ("noise_total", "lambda_reg", "lam_sig", "lambda_C", "alpha_AE", "alpha_C", "eta_clf", "gamma0", "gamma_mu", "h_scale", "ambient_dim")
    for name in scalar_param_names:
        values = np.asarray([item["params"][name] for item in flat_results], dtype=float)
        shared_param_grids[name] = values.reshape(nx, ny)

    theory_solver_grid = np.asarray([str(item["params"]["theory_solver"]) for item in flat_results]).reshape(nx, ny)
    eta_grid = np.asarray([item["params"]["eta"] for item in flat_results], dtype=float).reshape(nx, ny)
    g_grid = np.asarray([item["params"]["g"] for item in flat_results], dtype=float).reshape(nx, ny)

    save_dict: Dict[str, Any] = {
        "metric_names": np.asarray(PLOT_METRICS, dtype=str),
        "vary_x": np.asarray(sweep.vary_x),
        "vary_y": np.asarray(sweep.vary_y),
        "x_values": np.asarray(sweep.x_values),
        "y_values": np.asarray(sweep.y_values),
        "grid_shape": np.asarray([nx, ny], dtype=int),
        "config_json": np.asarray(config_json),
        "sweep_json": np.asarray(sweep_json),
        "measure_counts": measure_counts.reshape(nx, ny),
        "epoch_counts": epoch_counts.reshape(nx, ny),
        "theory_counts": theory_counts.reshape(nx, ny),
        "times": reshape_grid(times_flat, nx, ny),
        "times_sgd": reshape_grid(times_sgd_flat, nx, ny),
        "theory_times_dense": reshape_grid(theory_times_flat, nx, ny),
        "epoch_ae_losses": reshape_grid(ae_loss_flat, nx, ny),
        "epoch_clf_losses": reshape_grid(clf_loss_flat, nx, ny),
        "epoch_rec_losses": reshape_grid(rec_loss_flat, nx, ny),
        "gamma_history": reshape_grid(gamma_flat, nx, ny),
        "stable_reached": stable_reached,
        "stable_epoch": stable_epoch,
        "used_epochs": used_epochs,
        "used_steps": used_steps,
        "final_tau": final_tau,
        "final_rel_improve": final_rel_improve,
        "final_stability": final_stability,
        "matched_dt": matched_dt,
        "shared_eta": eta_grid,
        "shared_g": g_grid,
        "shared_theory_solver": theory_solver_grid,
    }

    for field_name, values in config_grids.items():
        save_dict[f"config_{field_name}"] = values

    for param_name, values in shared_param_grids.items():
        save_dict[f"shared_{param_name}"] = values

    for prefix, metric_dict in (
        ("measured", reshape_metric_grid(measured_metric_flat, nx, ny)),
        ("theory", reshape_metric_grid(theory_metric_flat, nx, ny)),
        ("theory_dense", reshape_metric_grid(theory_dense_metric_flat, nx, ny)),
    ):
        for metric_name, values in metric_dict.items():
            save_dict[f"{prefix}_{metric_name}"] = values

    ensure_parent_dir(sweep.out_data)
    np.savez(sweep.out_data, **save_dict)
    print(f"Saved grid results -> {sweep.out_data}")


if __name__ == "__main__":
    main()
