# Branch-B Fixed Point Report

## Scope

This report solves `F(x) = dx/dt = 0` for the Branch-B ODE implemented in `phase_diagram.py`. The search is numerical and multi-start; it reports distinct roots found under the configured tolerances, not a symbolic proof of global completeness.

## Runtime

- Rough estimate before solve: RHS eval ~= 0.096 ms; one central Jacobian classification ~= 0.013 s per fixed point. Root search rough range for 1 starts: 32 ms to 126 ms (typical mid estimate 126 ms).
- Actual elapsed time: 258 ms
- Starts attempted: `1`
- Distinct fixed points accepted: `1`

## Parameters

| parameter | value |
|---|---:|
| `noise_total` | `0.5` |
| `eta_fraction` | `0.5` |
| `eta` | `0.25` |
| `g` | `0.25` |
| `lambda_reg` | `0.5` |
| `lam_sig` | `15.0` |
| `lambda_C` | `0.1` |
| `alpha_AE` | `0.7` |
| `alpha_C` | `1.0` |
| `eta_clf` | `1.0` |
| `gamma0` | `0.0` |
| `gamma_mu` | `0.0` |
| `h_scale` | `0.2` |
| `ambient_dim` | `32.0` |
| `h_vec` | `[0.14142136 0.14142136 0.        ]` |
| frozen `tau` | `inf` |
| frozen `Gamma(tau)` | `1.0` |

## State Layout

The packed state has `65` variables with `D_DIM=3`, `R_DIM=3`.

| block | shape |
|---|---:|
| `M` | `3x3` |
| `s` | `3` |
| `N` | `3x3` |
| `a` | `3` |
| `beta` | `3` |
| `rho` | `scalar` |
| `C` | `3` |
| `Q` | `3x3` |
| `T` | `3x3` |
| `u` | `3` |
| `t` | `3` |
| `B` | `3x3` |
| `m` | `scalar` |

## Recovered Equations

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

## Classification Method

Dynamical behavior is classified using eigenvalues of the Jacobian `J = dF/dx` at each fixed point. Positive real parts are unstable directions, negative real parts are stable directions, and near-zero real parts are neutral/non-hyperbolic directions. The Hessian reported here is for `Phi(x) = 0.5 ||F(x)||^2`; at a true fixed point it is `J^T J`.

Eigenvalue tolerance: `1e-07`. Residual max tolerance: `0.01`.

## Fixed Point Summary

| id | classification | residual max | residual norm | positive | negative | neutral | Hessian zero modes | source start |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | saddle | 2.776e-17 | 3.925e-17 | 9 | 56 | 0 | 0 | 0 |

## Fixed Point 0

- Classification: `saddle`
- Residual norm: `3.925231146709e-17`
- Residual max abs: `2.775557561563e-17`
- Solver nfev: `3`
- Jacobian modes: positive `9`, negative `56`, neutral `0`
- Hessian zero modes: `0`

### Scalar Metrics

| metric | value |
|---|---:|
| `M_tilde` | `8.342592579955e-23` |
| `N_tilde` | `8.119630267760e-23` |
| `b_norm` | `3.333333333333e-01` |
| `b_perp_norm` | `0.000000000000e+00` |
| `latent_label_coupling` | `7.993993411850e-52` |
| `m` | `1.111111111111e-01` |
| `norm_B` | `6.447933729468e-24` |
| `norm_C` | `2.168806632024e-25` |
| `norm_M` | `8.342592579955e-27` |
| `norm_N` | `8.119630267760e-27` |
| `norm_Q` | `3.235296266859e-24` |
| `norm_T` | `3.218911062835e-24` |
| `norm_a` | `8.153036734924e-26` |
| `norm_beta` | `6.666666666667e-02` |
| `norm_s` | `9.537614696772e-26` |
| `norm_t` | `6.034682965909e-26` |
| `norm_u` | `3.944031752926e-26` |
| `reconstruction_error` | `5.311111111111e+01` |
| `rho` | `3.333333333333e-01` |

### State Blocks

`M`:

```text
[[ 5.27361600e-28 -2.61212386e-28 -5.36564083e-29]
 [ 5.14073650e-28  3.34974005e-27  3.01108496e-27]
 [ 6.78059228e-27  1.50589811e-27 -6.73651391e-28]]
```

`s`:

```text
[-1.31441791e-26 -3.97143641e-26  8.57123635e-26]
```

`N`:

```text
[[ 5.91997739e-28  1.04953899e-27  4.19643495e-27]
 [-2.27024385e-27  7.29465226e-30  3.95398271e-27]
 [ 3.13497484e-27 -4.02598399e-27  2.04161959e-28]]
```

`a`:

```text
[-1.11362550e-26 -7.59192956e-26 -2.75580330e-26]
```

`beta`:

```text
[ 4.71404521e-02  4.71404521e-02 -6.15461790e-26]
```

`rho`:

```text
0.33333333
```

`C`:

```text
[-4.87299842e-26 -9.71305451e-26 -1.87691950e-25]
```

`Q`:

```text
[[-1.68699555e-24  9.97203384e-27 -4.05040139e-26]
 [ 3.29929241e-26 -1.85678534e-24 -3.07669312e-27]
 [-2.45824180e-26  2.08981580e-27 -2.04207970e-24]]
```

`T`:

```text
[[-1.66198592e-24  1.51769747e-26 -1.91902279e-26]
 [ 1.32281959e-27 -1.85585908e-24  3.33339955e-27]
 [-1.89493264e-26  1.08260220e-27 -2.03813802e-24]]
```

`u`:

```text
[-8.34874607e-27 -3.49063786e-26 -1.63518139e-26]
```

`t`:

```text
[ 1.63774149e-28 -3.86101075e-26  4.63785794e-26]
```

`B`:

```text
[[-3.33070912e-24  3.57643804e-27 -4.56507705e-26]
 [ 3.94867469e-26 -3.72142619e-24  6.90353440e-29]
 [-1.51035032e-26  1.24749724e-27 -4.07790726e-24]]
```

`m`:

```text
0.11111111
```

### Jacobian Eigenvalues

Sorted by real part, descending:

```text
+2.065926e+01+0.000000e+00j, +2.065926e+01+0.000000e+00j, +2.065926e+01+0.000000e+00j, +2.065000e+01+0.000000e+00j, +2.065000e+01+0.000000e+00j, +2.065000e+01+0.000000e+00j, +2.065000e+01+0.000000e+00j, +2.065000e+01+0.000000e+00j, +2.065000e+01+0.000000e+00j, -2.021091e-01+2.903065e-01j, -2.021091e-01-2.903065e-01j, -2.021091e-01+2.903065e-01j, -2.021091e-01-2.903065e-01j, -2.021091e-01+2.903065e-01j, -2.021091e-01-2.903065e-01j, -7.000000e-01+0.000000e+00j, -7.000000e-01+0.000000e+00j, -7.000000e-01+0.000000e+00j, -7.000000e-01+0.000000e+00j, -7.000000e-01+0.000000e+00j, -7.000000e-01+0.000000e+00j, -1.050000e+00+0.000000e+00j, -1.050000e+00+0.000000e+00j, -1.050000e+00+0.000000e+00j, -1.050000e+00+0.000000e+00j, -1.195774e+00+0.000000e+00j, -1.195774e+00+0.000000e+00j, -1.195774e+00+0.000000e+00j, -1.400000e+00+0.000000e+00j, -1.400000e+00+6.984966e-10j, -1.400000e+00-6.984966e-10j, -1.400000e+00+0.000000e+00j, -1.400000e+00+0.000000e+00j, -1.400000e+00+0.000000e+00j, -1.400000e+00+0.000000e+00j, -1.400000e+00+0.000000e+00j, -1.400000e+00+0.000000e+00j, -1.400000e+00+0.000000e+00j, -1.400000e+00+0.000000e+00j, -1.400000e+00+0.000000e+00j, -1.400000e+00+0.000000e+00j, -1.400000e+00+1.562222e-15j, -1.400000e+00-1.562222e-15j, -1.400000e+00+6.985008e-10j, -1.400000e+00-6.985008e-10j, -1.400000e+00+0.000000e+00j, -2.100000e+00+0.000000e+00j, -2.100000e+00+0.000000e+00j, -2.100000e+00+0.000000e+00j, -2.100000e+00+0.000000e+00j, -2.100000e+00+0.000000e+00j, -2.100000e+00+0.000000e+00j, -2.100000e+00+6.661338e-16j, -2.100000e+00-6.661338e-16j, -2.100000e+00+0.000000e+00j, -2.100000e+00+0.000000e+00j, -2.205000e+01+0.000000e+00j, -2.205000e+01+0.000000e+00j, -2.205000e+01+0.000000e+00j, -2.205000e+01+0.000000e+00j, -2.205000e+01+0.000000e+00j, -2.205000e+01+0.000000e+00j, -2.205927e+01+0.000000e+00j, -2.205927e+01+0.000000e+00j, -2.205927e+01+0.000000e+00j
```

### Hessian Eigenvalues of Phi

Sorted ascending:

```text
5.843097e-02, 5.843097e-02, 5.843097e-02, 1.694137e-01, 1.694137e-01, 1.694137e-01, 4.613732e-01, 4.613732e-01, 4.613732e-01, 4.613732e-01, 4.613732e-01, 4.613732e-01, 9.652057e-01, 1.102500e+00, 1.102500e+00, 1.102500e+00, 1.378791e+00, 1.378791e+00, 1.378791e+00, 1.647104e+00, 1.647104e+00, 1.647104e+00, 1.960000e+00, 1.960000e+00, 1.960000e+00, 1.960000e+00, 1.960000e+00, 1.960000e+00, 1.960000e+00, 1.960000e+00, 1.960000e+00, 2.444278e+00, 2.444278e+00, 2.444278e+00, 2.786209e+00, 2.786209e+00, 2.786209e+00, 4.468479e+00, 4.468479e+00, 4.468479e+00, 4.683627e+00, 4.683627e+00, 4.683627e+00, 4.683627e+00, 4.683627e+00, 4.683627e+00, 5.037294e+00, 4.264225e+02, 4.264225e+02, 4.264225e+02, 4.264225e+02, 4.264225e+02, 4.264225e+02, 4.460724e+02, 4.460724e+02, 4.460724e+02, 4.862025e+02, 4.862025e+02, 4.862025e+02, 4.862025e+02, 4.862025e+02, 4.862025e+02, 5.059073e+02, 5.059073e+02, 5.059073e+02
```

## Solver Attempts

| start | accepted | distinct | residual max | residual norm | nfev | success | status |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | True | True | 2.776e-17 | 3.925e-17 | 3 | True | 1 |
