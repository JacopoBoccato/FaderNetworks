import numpy as np
import torch

from phase_diagram import D_DIM, R_DIM, pack_state, rhs, unpack_state
from src.branch_b_observables import compute_branch_b_time_derivatives


def _torch_observables_from_state(state, params):
    M, s, N, a, beta, rho, C, Q, T, u, t, B, m = unpack_state(state)
    dtype = torch.float64

    obs = {
        "M": torch.tensor(M, dtype=dtype),
        "s": torch.tensor(s, dtype=dtype),
        "N": torch.tensor(N, dtype=dtype),
        "a": torch.tensor(a, dtype=dtype),
        "beta": torch.tensor(beta, dtype=dtype),
        "rho": torch.tensor(rho, dtype=dtype),
        "C": torch.tensor(C, dtype=dtype),
        "Q": torch.tensor(Q, dtype=dtype),
        "T": torch.tensor(T, dtype=dtype),
        "u": torch.tensor(u, dtype=dtype),
        "t": torch.tensor(t, dtype=dtype),
        "B": torch.tensor(B, dtype=dtype),
        "m": torch.tensor(m, dtype=dtype),
    }

    lam_sig = float(params["lam_sig"])
    g = float(params["g"])
    eta = float(params["eta"])
    Lambda = lam_sig * torch.eye(R_DIM, dtype=dtype)
    obs["D"] = (lam_sig + eta) * torch.eye(R_DIM, dtype=dtype)
    obs["kappa"] = torch.tensor(g + eta, dtype=dtype)
    obs["g"] = torch.tensor(g, dtype=dtype)
    obs["eta"] = torch.tensor(eta, dtype=dtype)
    obs["S"] = obs["M"] @ Lambda @ obs["M"].t() + g * torch.outer(obs["s"], obs["s"]) + eta * obs["Q"]
    obs["G"] = obs["N"].t() @ Lambda @ obs["M"].t() + g * torch.outer(obs["a"], obs["s"]) + eta * obs["B"]
    obs["J"] = obs["N"].t() @ Lambda @ obs["N"] + g * torch.outer(obs["a"], obs["a"]) + eta * obs["T"]
    obs["H"] = obs["M"] @ Lambda @ obs["beta"] + g * obs["rho"] * obs["s"] + eta * obs["t"]
    obs["q"] = obs["N"].t() @ Lambda @ obs["beta"] + g * obs["rho"] * obs["a"] + eta * obs["u"]
    return obs


def _pack_torch_derivatives(derivs):
    return pack_state(
        derivs["dM_dt"].numpy(),
        derivs["ds_dt"].numpy(),
        derivs["dN_dt"].numpy(),
        derivs["da_dt"].numpy(),
        derivs["dbeta_dt"].numpy(),
        float(derivs["drho_dt"].item()),
        derivs["dC_dt"].numpy(),
        derivs["dQ_dt"].numpy(),
        derivs["dT_dt"].numpy(),
        derivs["du_dt"].numpy(),
        derivs["dt_dt"].numpy(),
        derivs["dB_dt"].numpy(),
        float(derivs["dm_dt"].item()),
    )


def test_torch_branch_b_derivatives_match_numpy_rhs():
    rng = np.random.default_rng(4)
    state = rng.normal(scale=0.15, size=65)
    params = {
        "lam_sig": 2.5,
        "g": 0.35,
        "eta": 0.2,
        "lambda_reg": 0.17,
        "lambda_C": 0.09,
        "alpha_AE": 0.7,
        "alpha_C": 1.3,
        "eta_clf": 1.1,
        "gamma0": 1.1,
        "gamma_mu": 0.0,
        "h_vec": np.array([0.12, -0.04, 0.02], dtype=float),
        "ambient_dim": 32,
    }

    numpy_rhs = rhs(0.0, state, params)
    obs = _torch_observables_from_state(state, params)
    derivs = compute_branch_b_time_derivatives(
        obs,
        params["lambda_reg"],
        params["lambda_reg"],
        params["lambda_reg"],
        params["lambda_C"],
        h=torch.tensor(params["h_vec"], dtype=torch.float64),
        gamma=params["eta_clf"],
        alpha_AE=params["alpha_AE"],
        alpha_C=params["alpha_C"],
    )

    torch_rhs = _pack_torch_derivatives(derivs)
    assert np.allclose(torch_rhs, numpy_rhs, rtol=1e-10, atol=1e-10)
