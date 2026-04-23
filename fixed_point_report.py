#!/usr/bin/env python3
"""
Find and classify fixed points of the Branch-B observable ODE.

The script reuses the ODE implemented in `phase_diagram.py`, searches for roots
of F(x) = dx/dt, and writes a Markdown report plus a JSON sidecar.

Notes:
- This is a numerical multi-start root search in the full 65-dimensional state.
  It reports every distinct fixed point it finds, but no finite numerical search
  can prove global completeness when the vector field has symmetries or distant
  roots.
- Dynamical behavior is classified from the Jacobian eigenvalues. Because the
  Hessian of a vector field is a third-order tensor, the report also includes
  the Hessian of Phi(x) = 0.5 ||F(x)||^2. At a true fixed point this Hessian is
  J^T J, so its near-zero modes flag flat or non-isolated directions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from phase_diagram import (
    D_DIM,
    H_DIRECTION,
    N_DIM,
    R_DIM,
    classifier_strength,
    pack_state,
    rhs,
    scalarize_state,
    unpack_state,
)


STATE_BLOCKS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("M", (D_DIM, R_DIM)),
    ("s", (D_DIM,)),
    ("N", (R_DIM, D_DIM)),
    ("a", (D_DIM,)),
    ("beta", (R_DIM,)),
    ("rho", ()),
    ("C", (D_DIM,)),
    ("Q", (D_DIM, D_DIM)),
    ("T", (D_DIM, D_DIM)),
    ("u", (D_DIM,)),
    ("t", (D_DIM,)),
    ("B", (D_DIM, D_DIM)),
    ("m", ()),
)
STATE_DIM = (
    D_DIM * R_DIM
    + D_DIM
    + R_DIM * D_DIM
    + D_DIM
    + R_DIM
    + 1
    + D_DIM
    + D_DIM * D_DIM
    + D_DIM * D_DIM
    + D_DIM
    + D_DIM
    + D_DIM * D_DIM
    + 1
)


@dataclass
class FixedPoint:
    index: int
    state: np.ndarray
    residual_norm: float
    residual_max_abs: float
    source_start: int
    nfev: int
    cost: float
    solver_success: bool
    solver_status: int
    solver_message: str
    metrics: Dict[str, float]
    jacobian_eigenvalues: np.ndarray
    hessian_eigenvalues: np.ndarray
    classification: str
    positive_modes: int
    negative_modes: int
    neutral_modes: int
    hessian_zero_modes: int


def parse_vec3(raw: str) -> np.ndarray:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("--h_vec must contain exactly 3 comma-separated numbers.")
    return np.asarray(values, dtype=float)


def parse_tau(raw: str) -> float:
    if raw.lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    return float(raw)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Numerically find and classify Branch-B ODE fixed points."
    )

    parser.add_argument("--noise_total", type=float, default=0.5)
    parser.add_argument(
        "--eta_fraction",
        type=float,
        default=0.5,
        help="Fraction of noise_total assigned to eta when --eta/--g are not set.",
    )
    parser.add_argument("--eta", type=float, default=None, help="Override eta directly.")
    parser.add_argument("--g", type=float, default=None, help="Override g directly.")
    parser.add_argument("--lambda_reg", type=float, default=0.5)
    parser.add_argument("--lam_sig", type=float, default=15.0)
    parser.add_argument("--lambda_C", type=float, default=0.1)
    parser.add_argument("--alpha_AE", type=float, default=1.0)
    parser.add_argument("--alpha_C", type=float, default=1.0)
    parser.add_argument("--eta_clf", type=float, default=1.0, help="Asymptotic classifier strength.")
    parser.add_argument("--gamma0", type=float, default=0.0)
    parser.add_argument("--gamma_mu", type=float, default=0.0)
    parser.add_argument("--h_scale", type=float, default=0.2)
    parser.add_argument(
        "--h_vec",
        type=parse_vec3,
        default=None,
        help="Explicit 3-vector h, e.g. '0.1,0.1,0.0'. Overrides --h_scale.",
    )
    parser.add_argument("--ambient_dim", type=float, default=float(N_DIM))
    parser.add_argument(
        "--tau",
        type=parse_tau,
        default=float("inf"),
        help="Freeze the vector field at this tau. Use 'inf' for the asymptotic system.",
    )

    parser.add_argument("--n_starts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--random_scale",
        type=float,
        default=0.25,
        help="Base standard deviation for random initial states.",
    )
    parser.add_argument(
        "--scale_multipliers",
        type=str,
        default="0.1,0.3,1,3",
        help="Comma-separated multipliers applied cyclically to --random_scale.",
    )
    parser.add_argument("--max_nfev", type=int, default=8000)
    parser.add_argument("--method", choices=("trf", "lm"), default="trf")
    parser.add_argument("--residual_tol", type=float, default=1e-7)
    parser.add_argument("--dedup_tol", type=float, default=1e-5)
    parser.add_argument("--eig_tol", type=float, default=1e-7)
    parser.add_argument("--jac_step", type=float, default=1e-5)
    parser.add_argument("--no_origin_start", action="store_true")
    parser.add_argument("--progress_every", type=int, default=5)
    parser.add_argument(
        "--estimate_only",
        action="store_true",
        help="Print a rough runtime estimate and exit before root search.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("results", "branch_b_fixed_point_report.md"),
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="JSON sidecar path. Defaults to the Markdown path with .json extension.",
    )
    return parser


def build_params(args: argparse.Namespace) -> Dict[str, Any]:
    eta_fraction = float(args.eta_fraction)
    if not 0.0 <= eta_fraction <= 1.0:
        raise ValueError("--eta_fraction must be in [0, 1].")

    eta = float(args.eta) if args.eta is not None else max(eta_fraction * args.noise_total, 1e-12)
    g = float(args.g) if args.g is not None else max((1.0 - eta_fraction) * args.noise_total, 1e-12)
    h_vec = np.asarray(args.h_vec, dtype=float) if args.h_vec is not None else float(args.h_scale) * H_DIRECTION

    return {
        "noise_total": float(args.noise_total),
        "eta_fraction": eta_fraction,
        "lambda_reg": float(args.lambda_reg),
        "lam_sig": float(args.lam_sig),
        "lambda_C": float(args.lambda_C),
        "alpha_AE": max(float(args.alpha_AE), 0.0),
        "alpha_C": max(float(args.alpha_C), 0.0),
        "eta_clf": float(args.eta_clf),
        "gamma0": float(args.gamma0),
        "gamma_mu": float(args.gamma_mu),
        "h_scale": float(args.h_scale),
        "h_vec": h_vec,
        "ambient_dim": float(args.ambient_dim),
        "eta": eta,
        "g": g,
    }


def state_to_blocks(state: np.ndarray) -> Dict[str, Any]:
    M, s, N, a, beta, rho, C, Q, T, u, t, B, m = unpack_state(state)
    return {
        "M": M,
        "s": s,
        "N": N,
        "a": a,
        "beta": beta,
        "rho": float(rho),
        "C": C,
        "Q": Q,
        "T": T,
        "u": u,
        "t": t,
        "B": B,
        "m": float(m),
    }


def zero_state() -> np.ndarray:
    return pack_state(
        np.zeros((D_DIM, R_DIM)),
        np.zeros(D_DIM),
        np.zeros((R_DIM, D_DIM)),
        np.zeros(D_DIM),
        np.zeros(R_DIM),
        0.0,
        np.zeros(D_DIM),
        np.zeros((D_DIM, D_DIM)),
        np.zeros((D_DIM, D_DIM)),
        np.zeros(D_DIM),
        np.zeros(D_DIM),
        np.zeros((D_DIM, D_DIM)),
        0.0,
    )


def structured_state(params: Dict[str, Any]) -> np.ndarray:
    h = np.asarray(params["h_vec"], dtype=float)
    lambda_reg = float(params["lambda_reg"])
    g = float(params["g"])
    rho = g / max(g + lambda_reg, 1e-12)
    beta = (g / max(g + lambda_reg, 1e-12)) * h
    s = h.copy()
    a = h.copy()
    C = (g / max(float(params["lambda_C"]) + g, 1e-12)) * h
    Q = np.eye(D_DIM)
    T = np.eye(D_DIM)
    B = np.eye(D_DIM)
    return pack_state(
        np.eye(D_DIM, R_DIM),
        s,
        np.eye(R_DIM, D_DIM),
        a,
        beta,
        rho,
        C,
        Q,
        T,
        np.zeros(D_DIM),
        np.zeros(D_DIM),
        B,
        rho * rho,
    )


def parse_scale_multipliers(raw: str) -> Tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("--scale_multipliers must contain at least one value.")
    return values


def make_initial_states(args: argparse.Namespace, params: Dict[str, Any]) -> List[np.ndarray]:
    rng = np.random.default_rng(args.seed)
    starts: List[np.ndarray] = []
    if not args.no_origin_start:
        starts.append(zero_state())
        starts.append(structured_state(params))

    multipliers = parse_scale_multipliers(args.scale_multipliers)
    while len(starts) < int(args.n_starts):
        mult = multipliers[(len(starts) - 1) % len(multipliers)]
        starts.append(rng.normal(0.0, float(args.random_scale) * mult, size=STATE_DIM))
    return starts[: int(args.n_starts)]


def residual_function(params: Dict[str, Any], tau: float) -> Callable[[np.ndarray], np.ndarray]:
    def fun(state: np.ndarray) -> np.ndarray:
        values = rhs(tau, state, params)
        if not np.all(np.isfinite(values)):
            return np.full_like(values, 1e100, dtype=float)
        return values

    return fun


def benchmark_rhs(fun: Callable[[np.ndarray], np.ndarray], sample: np.ndarray, n_eval: int = 200) -> float:
    start = time.perf_counter()
    for _ in range(n_eval):
        fun(sample)
    elapsed = time.perf_counter() - start
    return elapsed / max(n_eval, 1)


def rough_runtime_text(rhs_seconds: float, args: argparse.Namespace, state_dim: int) -> str:
    jac_seconds = (2 * state_dim + 1) * rhs_seconds
    optimizer_step_seconds = (state_dim + 1) * rhs_seconds
    lower_steps = min(5, max(args.max_nfev, 1))
    mid_steps = min(50, max(args.max_nfev, 1))
    upper_steps = min(1000, max(args.max_nfev, 1))
    lower = max(args.n_starts, 1) * lower_steps * optimizer_step_seconds
    mid = max(args.n_starts, 1) * mid_steps * optimizer_step_seconds
    upper = max(args.n_starts, 1) * upper_steps * optimizer_step_seconds
    return (
        f"RHS eval ~= {rhs_seconds * 1e3:.3f} ms; one central Jacobian classification "
        f"~= {jac_seconds:.3f} s per fixed point. Root search rough range for "
        f"{args.n_starts} starts: {format_seconds(lower)} to {format_seconds(upper)} "
        f"(typical mid estimate {format_seconds(mid)})."
    )


def format_seconds(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds / 3600.0:.1f} h"


def is_distinct(candidate: np.ndarray, points: Sequence[FixedPoint], tol: float) -> bool:
    if not points:
        return True
    scale = math.sqrt(candidate.size)
    for point in points:
        dist = float(np.linalg.norm(candidate - point.state) / max(scale, 1.0))
        if dist <= tol:
            return False
    return True


def finite_difference_jacobian(
    fun: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    rel_step: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    f0 = fun(x)
    jac = np.empty((f0.size, x.size), dtype=float)
    for j in range(x.size):
        step = rel_step * max(1.0, abs(float(x[j])))
        xp = x.copy()
        xm = x.copy()
        xp[j] += step
        xm[j] -= step
        jac[:, j] = (fun(xp) - fun(xm)) / (2.0 * step)
    return jac


def classify_from_jacobian(eigs: np.ndarray, eig_tol: float) -> Tuple[str, int, int, int]:
    real_parts = np.real(eigs)
    pos = int(np.sum(real_parts > eig_tol))
    neg = int(np.sum(real_parts < -eig_tol))
    neutral = int(eigs.size - pos - neg)

    if pos == 0 and neutral == 0:
        label = "asymptotically stable attractor"
    elif neg == 0 and neutral == 0:
        label = "unstable source"
    elif pos > 0 and neg > 0:
        label = "saddle"
    elif pos > 0 and neg == 0:
        label = "non-hyperbolic unstable"
    elif pos == 0 and neg > 0:
        label = "non-hyperbolic stable/center-manifold"
    else:
        label = "fully non-hyperbolic/center-like"
    return label, pos, neg, neutral


def analyze_fixed_point(
    index: int,
    state: np.ndarray,
    residual: np.ndarray,
    result: Any,
    source_start: int,
    fun: Callable[[np.ndarray], np.ndarray],
    params: Dict[str, Any],
    args: argparse.Namespace,
) -> FixedPoint:
    jac = finite_difference_jacobian(fun, state, float(args.jac_step))
    jac_eigs = np.linalg.eigvals(jac)
    hessian_phi = jac.T @ jac
    hessian_eigs = np.linalg.eigvalsh(hessian_phi)
    classification, pos, neg, neutral = classify_from_jacobian(jac_eigs, float(args.eig_tol))
    h_zero = int(np.sum(np.abs(hessian_eigs) <= float(args.eig_tol)))
    return FixedPoint(
        index=index,
        state=state,
        residual_norm=float(np.linalg.norm(residual)),
        residual_max_abs=float(np.max(np.abs(residual))),
        source_start=source_start,
        nfev=int(result.nfev),
        cost=float(result.cost),
        solver_success=bool(result.success),
        solver_status=int(result.status),
        solver_message=str(result.message),
        metrics=scalarize_state(state, params),
        jacobian_eigenvalues=jac_eigs,
        hessian_eigenvalues=hessian_eigs,
        classification=classification,
        positive_modes=pos,
        negative_modes=neg,
        neutral_modes=neutral,
        hessian_zero_modes=h_zero,
    )


def find_fixed_points(
    starts: Sequence[np.ndarray],
    fun: Callable[[np.ndarray], np.ndarray],
    params: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[List[FixedPoint], List[Dict[str, Any]]]:
    fixed_points: List[FixedPoint] = []
    attempts: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    for start_index, x0 in enumerate(starts):
        result = least_squares(
            fun,
            x0,
            method=str(args.method),
            max_nfev=int(args.max_nfev),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            x_scale="jac",
        )
        residual = fun(result.x)
        residual_norm = float(np.linalg.norm(residual))
        residual_max_abs = float(np.max(np.abs(residual)))
        accepted = residual_max_abs <= float(args.residual_tol)
        distinct = accepted and is_distinct(result.x, fixed_points, float(args.dedup_tol))

        attempts.append(
            {
                "start": start_index,
                "accepted": bool(accepted),
                "distinct": bool(distinct),
                "residual_norm": residual_norm,
                "residual_max_abs": residual_max_abs,
                "nfev": int(result.nfev),
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
            }
        )

        if distinct:
            fixed_points.append(
                analyze_fixed_point(
                    len(fixed_points),
                    result.x.copy(),
                    residual,
                    result,
                    start_index,
                    fun,
                    params,
                    args,
                )
            )

        progress_every = max(int(args.progress_every), 0)
        if progress_every and (start_index + 1) % progress_every == 0:
            elapsed = time.perf_counter() - start_time
            avg = elapsed / float(start_index + 1)
            total_est = avg * len(starts)
            print(
                f"[{start_index + 1}/{len(starts)}] elapsed {format_seconds(elapsed)}, "
                f"projected total {format_seconds(total_est)}, distinct fixed points {len(fixed_points)}",
                flush=True,
            )

    return fixed_points, attempts


def array_to_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return array_to_json(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return "nan"
    if isinstance(value, np.generic):
        return array_to_json(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {key: array_to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [array_to_json(item) for item in value]
    return value


def fixed_point_to_json(point: FixedPoint) -> Dict[str, Any]:
    return {
        "index": point.index,
        "residual_norm": point.residual_norm,
        "residual_max_abs": point.residual_max_abs,
        "source_start": point.source_start,
        "nfev": point.nfev,
        "cost": point.cost,
        "solver_success": point.solver_success,
        "solver_status": point.solver_status,
        "solver_message": point.solver_message,
        "classification": point.classification,
        "positive_modes": point.positive_modes,
        "negative_modes": point.negative_modes,
        "neutral_modes": point.neutral_modes,
        "hessian_zero_modes": point.hessian_zero_modes,
        "metrics": point.metrics,
        "state": state_to_blocks(point.state),
        "jacobian_eigenvalues": [
            {"real": float(val.real), "imag": float(val.imag)} for val in point.jacobian_eigenvalues
        ],
        "hessian_eigenvalues": point.hessian_eigenvalues,
    }


def format_matrix(value: Any, precision: int = 8) -> str:
    arr = np.asarray(value)
    return np.array2string(arr, precision=precision, suppress_small=False, max_line_width=140)


def format_complex_eigs(eigs: np.ndarray, limit: Optional[int] = None) -> str:
    ordered = sorted(eigs, key=lambda z: (float(np.real(z)), float(np.imag(z))), reverse=True)
    if limit is not None:
        ordered = ordered[:limit]
    parts = []
    for val in ordered:
        parts.append(f"{val.real:+.6e}{val.imag:+.6e}j")
    return ", ".join(parts)


def format_real_eigs(eigs: np.ndarray, limit: Optional[int] = None) -> str:
    ordered = np.sort(np.asarray(eigs, dtype=float))
    if limit is not None and ordered.size > limit:
        left = ordered[: limit // 2]
        right = ordered[-(limit - left.size) :]
        ordered = np.concatenate([left, right])
    return ", ".join(f"{val:.6e}" for val in ordered)


def equations_markdown() -> str:
    return r"""
Definitions:

```text
Lambda = lam_sig I_R
D      = (lam_sig + eta) I_R
kappa  = g + eta
CCt    = C C^T

S = M Lambda M^T + g s s^T + eta Q
G = N^T Lambda M^T + g a s^T + eta B
J = N^T Lambda N + g a a^T + eta T
H = M Lambda beta + g rho s + eta t
q = N^T Lambda beta + g rho a + eta u
aux = T S - G + g u s^T
```

Fixed point equations, i.e. set every derivative below to zero:

```text
0 = dM =
    -2 alpha_AE (T M D - N^T D + g (T s - a + u) h^T)
    +2 alpha_AE Gamma (CCt (M D + g s h^T))
    -2 alpha_AE Gamma g C h^T
    -2 alpha_AE lambda_reg M

0 = ds =
    -2 alpha_AE ((T M - N^T) Lambda h + kappa T s - kappa a + g u)
    +2 alpha_AE Gamma (CCt (M Lambda h + kappa s))
    -2 alpha_AE Gamma g C
    -2 alpha_AE lambda_reg s

0 = dN =
    -2 alpha_AE (N S - D M^T + g (beta - h) s^T)
    -2 alpha_AE lambda_reg N

0 = da =
    -2 alpha_AE (S a - M Lambda h - kappa s + g rho s)
    -2 alpha_AE lambda_reg a

0 = dbeta =
    -2 alpha_AE g (N s - h + beta)
    -2 alpha_AE lambda_reg beta

0 = drho =
    -2 alpha_AE g (a^T s - 1 + rho)
    -2 alpha_AE lambda_reg rho

0 = dC =
    -2 alpha_C (Gamma (S C - g s) + lambda_C C)

0 = dQ =
    -2 alpha_AE aux - 2 alpha_AE aux^T
    +2 alpha_AE Gamma (CCt S + S CCt)
    -2 alpha_AE Gamma g (C s^T + s C^T)
    -4 alpha_AE lambda_reg Q

0 = dT =
    -2 alpha_AE aux - 2 alpha_AE aux^T
    -4 alpha_AE lambda_reg T

0 = du =
    -2 alpha_AE (S u - H + g m s)
    -2 alpha_AE g (T s - a + u)
    -2 alpha_AE (lambda_reg + lambda_reg) u

0 = dt =
    -2 alpha_AE (T H - q + g rho u)
    +2 alpha_AE Gamma (CCt H)
    -2 alpha_AE Gamma g rho C
    -2 alpha_AE g (B^T s - s + t)
    -2 alpha_AE (lambda_reg + lambda_reg) t

0 = dB =
    -2 alpha_AE (S B - S + g s t^T)
    -2 alpha_AE (G T - J + g a u^T)
    +2 alpha_AE Gamma (G CCt)
    -2 alpha_AE Gamma g a C^T
    -2 alpha_AE (lambda_reg + lambda_reg) B

0 = dm =
    -4 alpha_AE g (u^T s - rho + m)
    -4 alpha_AE lambda_reg m
```
""".strip()


def params_markdown(params: Dict[str, Any], tau: float) -> str:
    gamma = classifier_strength(tau, params)
    lines = [
        "| parameter | value |",
        "|---|---:|",
    ]
    for key in (
        "noise_total",
        "eta_fraction",
        "eta",
        "g",
        "lambda_reg",
        "lam_sig",
        "lambda_C",
        "alpha_AE",
        "alpha_C",
        "eta_clf",
        "gamma0",
        "gamma_mu",
        "h_scale",
        "ambient_dim",
    ):
        value = params[key]
        lines.append(f"| `{key}` | `{value}` |")
    lines.append(f"| `h_vec` | `{format_matrix(params['h_vec'])}` |")
    lines.append(f"| frozen `tau` | `{tau}` |")
    lines.append(f"| frozen `Gamma(tau)` | `{gamma}` |")
    return "\n".join(lines)


def state_layout_markdown() -> str:
    lines = [
        f"The packed state has `{STATE_DIM}` variables with `D_DIM={D_DIM}`, `R_DIM={R_DIM}`.",
        "",
        "| block | shape |",
        "|---|---:|",
    ]
    for name, shape in STATE_BLOCKS:
        shape_text = "scalar" if not shape else "x".join(str(item) for item in shape)
        lines.append(f"| `{name}` | `{shape_text}` |")
    return "\n".join(lines)


def write_markdown_report(
    path: str,
    params: Dict[str, Any],
    tau: float,
    args: argparse.Namespace,
    runtime_estimate: str,
    elapsed: float,
    fixed_points: Sequence[FixedPoint],
    attempts: Sequence[Dict[str, Any]],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Branch-B Fixed Point Report\n\n")
        handle.write("## Scope\n\n")
        handle.write(
            "This report solves `F(x) = dx/dt = 0` for the Branch-B ODE implemented in "
            "`phase_diagram.py`. The search is numerical and multi-start; it reports "
            "distinct roots found under the configured tolerances, not a symbolic proof "
            "of global completeness.\n\n"
        )
        if float(params["gamma_mu"]) > 0.0:
            handle.write(
                "Because `gamma_mu > 0`, the original ODE is non-autonomous. Fixed points "
                "below are for the vector field frozen at the selected `tau`.\n\n"
            )
        handle.write("## Runtime\n\n")
        handle.write(f"- Rough estimate before solve: {runtime_estimate}\n")
        handle.write(f"- Actual elapsed time: {format_seconds(elapsed)}\n")
        handle.write(f"- Starts attempted: `{len(attempts)}`\n")
        handle.write(f"- Distinct fixed points accepted: `{len(fixed_points)}`\n\n")

        handle.write("## Parameters\n\n")
        handle.write(params_markdown(params, tau))
        handle.write("\n\n")

        handle.write("## State Layout\n\n")
        handle.write(state_layout_markdown())
        handle.write("\n\n")

        handle.write("## Recovered Equations\n\n")
        handle.write(equations_markdown())
        handle.write("\n\n")

        handle.write("## Classification Method\n\n")
        handle.write(
            "Dynamical behavior is classified using eigenvalues of the Jacobian `J = dF/dx` "
            "at each fixed point. Positive real parts are unstable directions, negative real "
            "parts are stable directions, and near-zero real parts are neutral/non-hyperbolic "
            "directions. The Hessian reported here is for `Phi(x) = 0.5 ||F(x)||^2`; at a "
            "true fixed point it is `J^T J`.\n\n"
        )
        handle.write(f"Eigenvalue tolerance: `{args.eig_tol}`. Residual max tolerance: `{args.residual_tol}`.\n\n")

        handle.write("## Fixed Point Summary\n\n")
        if fixed_points:
            handle.write(
                "| id | classification | residual max | residual norm | positive | negative | neutral | Hessian zero modes | source start |\n"
            )
            handle.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for point in fixed_points:
                handle.write(
                    f"| {point.index} | {point.classification} | {point.residual_max_abs:.3e} | "
                    f"{point.residual_norm:.3e} | {point.positive_modes} | {point.negative_modes} | "
                    f"{point.neutral_modes} | {point.hessian_zero_modes} | {point.source_start} |\n"
                )
            handle.write("\n")
        else:
            handle.write("No fixed points met the configured residual tolerance.\n\n")

        for point in fixed_points:
            handle.write(f"## Fixed Point {point.index}\n\n")
            handle.write(f"- Classification: `{point.classification}`\n")
            handle.write(f"- Residual norm: `{point.residual_norm:.12e}`\n")
            handle.write(f"- Residual max abs: `{point.residual_max_abs:.12e}`\n")
            handle.write(f"- Solver nfev: `{point.nfev}`\n")
            handle.write(
                f"- Jacobian modes: positive `{point.positive_modes}`, negative `{point.negative_modes}`, "
                f"neutral `{point.neutral_modes}`\n"
            )
            handle.write(f"- Hessian zero modes: `{point.hessian_zero_modes}`\n\n")

            handle.write("### Scalar Metrics\n\n")
            handle.write("| metric | value |\n|---|---:|\n")
            for key, value in sorted(point.metrics.items()):
                handle.write(f"| `{key}` | `{value:.12e}` |\n")
            handle.write("\n")

            handle.write("### State Blocks\n\n")
            blocks = state_to_blocks(point.state)
            for name in blocks:
                handle.write(f"`{name}`:\n\n")
                handle.write("```text\n")
                handle.write(format_matrix(blocks[name]))
                handle.write("\n```\n\n")

            handle.write("### Jacobian Eigenvalues\n\n")
            handle.write("Sorted by real part, descending:\n\n")
            handle.write("```text\n")
            handle.write(format_complex_eigs(point.jacobian_eigenvalues))
            handle.write("\n```\n\n")

            handle.write("### Hessian Eigenvalues of Phi\n\n")
            handle.write("Sorted ascending:\n\n")
            handle.write("```text\n")
            handle.write(format_real_eigs(point.hessian_eigenvalues))
            handle.write("\n```\n\n")

        handle.write("## Solver Attempts\n\n")
        handle.write("| start | accepted | distinct | residual max | residual norm | nfev | success | status |\n")
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for attempt in attempts:
            handle.write(
                f"| {attempt['start']} | {attempt['accepted']} | {attempt['distinct']} | "
                f"{attempt['residual_max_abs']:.3e} | {attempt['residual_norm']:.3e} | "
                f"{attempt['nfev']} | {attempt['success']} | {attempt['status']} |\n"
            )


def write_json_report(
    path: str,
    params: Dict[str, Any],
    tau: float,
    args: argparse.Namespace,
    runtime_estimate: str,
    elapsed: float,
    fixed_points: Sequence[FixedPoint],
    attempts: Sequence[Dict[str, Any]],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "state_dim": STATE_DIM,
        "D_DIM": D_DIM,
        "R_DIM": R_DIM,
        "tau": tau,
        "gamma_tau": classifier_strength(tau, params),
        "params": params,
        "search": {
            "n_starts": args.n_starts,
            "seed": args.seed,
            "random_scale": args.random_scale,
            "scale_multipliers": args.scale_multipliers,
            "max_nfev": args.max_nfev,
            "method": args.method,
            "residual_tol": args.residual_tol,
            "dedup_tol": args.dedup_tol,
            "eig_tol": args.eig_tol,
            "jac_step": args.jac_step,
        },
        "runtime_estimate": runtime_estimate,
        "elapsed_seconds": elapsed,
        "fixed_points": [fixed_point_to_json(point) for point in fixed_points],
        "attempts": attempts,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(array_to_json(data), handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.n_starts <= 0:
        raise ValueError("--n_starts must be positive.")
    if args.max_nfev <= 0:
        raise ValueError("--max_nfev must be positive.")

    params = build_params(args)
    tau = float(args.tau)
    fun = residual_function(params, tau)
    starts = make_initial_states(args, params)

    rhs_seconds = benchmark_rhs(fun, starts[0])
    runtime_estimate = rough_runtime_text(rhs_seconds, args, STATE_DIM)
    print(runtime_estimate, flush=True)
    if args.estimate_only:
        return

    print(
        f"Searching {len(starts)} starts in {STATE_DIM} variables at tau={tau} "
        f"(Gamma={classifier_strength(tau, params):.6g}).",
        flush=True,
    )
    start_time = time.perf_counter()
    fixed_points, attempts = find_fixed_points(starts, fun, params, args)
    elapsed = time.perf_counter() - start_time

    fixed_points = sorted(fixed_points, key=lambda point: (point.classification, point.residual_max_abs))
    for new_index, point in enumerate(fixed_points):
        point.index = new_index

    out_json = args.out_json
    if out_json is None:
        root, _ = os.path.splitext(args.out)
        out_json = f"{root}.json"

    write_markdown_report(args.out, params, tau, args, runtime_estimate, elapsed, fixed_points, attempts)
    write_json_report(out_json, params, tau, args, runtime_estimate, elapsed, fixed_points, attempts)

    print(
        f"Done in {format_seconds(elapsed)}. Found {len(fixed_points)} distinct fixed points. "
        f"Report: {args.out}. JSON: {out_json}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
