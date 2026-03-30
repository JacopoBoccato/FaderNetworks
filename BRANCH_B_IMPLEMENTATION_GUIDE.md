# Branch B Implementation Guide: Theory to Code Mapping

## 1. Overview of Branch B Closed System

From the PDF (Section 4.2), the Branch B route defines a **13-dimensional macroscopic state** that evolves deterministically:

**Core Projected Observables:**
- $M \in \mathbb{R}^{d \times r}$: Encoder-signal projection ($M = WU$)
- $s \in \mathbb{R}^{d}$: Encoder-nuisance projection ($s = Wv$)
- $N \in \mathbb{R}^{r \times d}$: Decoder-signal projection ($N = U^T A$)
- $a \in \mathbb{R}^{d}$: Decoder-nuisance projection ($a = A^T v$)
- $\beta \in \mathbb{R}^{r}$: Bias-signal projection ($\beta = U^T b$)
- $\rho \in \mathbb{R}$: Bias-nuisance projection ($\rho = v^T b$)
- $C \in \mathbb{R}^{d}$: Classifier vector

**Bulk Observables** (for closure):
- $Q \in \mathbb{R}^{d \times d}$: $Q = WW^T$ (encoder autocorrelation)
- $T \in \mathbb{R}^{d \times d}$: $T = A^T A$ (decoder gram matrix)
- $u \in \mathbb{R}^{d}$: $u = A^T b$ (decoder-bias inner product)
- $t \in \mathbb{R}^{d}$: $t = Wb$ (encoder-bias product)
- $B \in \mathbb{R}^{d \times d}$: $B = A^T W^T$ (decoder-encoder correlation)
- $m \in \mathbb{R}$: $m = b^T b$ (bias squared norm)

## 2. The 13 Coupled Differential Equations (Eqs. 155-167)

### Auxiliary Definitions (Static Computations)
```
S = M Λ M^T + g ss^T + ηQ                    [Eq. 150: (d × d)]
G = N^T Λ M^T + g as^T + ηB                 [Eq. 151: (d × d)]
J = N^T Λ N + g aa^T + ηT                   [Eq. 152: (d × d)]
H = M Λ β + gρ s + ηt                       [Eq. 153: (d)]
q = N^T Λ β + gρ a + ηu                     [Eq. 154: (d)]
```

Where:
- $g = \sigma_y^2$ (teacher output variance)
- $\eta$ = isotropic noise level
- $\Lambda = \text{diag}(\lambda_1, ..., \lambda_r)$ = signal eigenvalues

### The 13 Differential Equations

**1. Encoder-Signal Evolution** (Eq. 155)
$$\dot{M} = -2(TMD - N^T D) + 2CC^T MD - 2\lambda_W M$$
where $D = \Lambda + \eta I_r$

**2. Encoder-Nuisance Evolution** (Eq. 156)
$$\dot{s} = -2(\kappa T s - \kappa a + gu) + 2\kappa CC^T s - 2gC - 2\lambda_W s$$
where $\kappa = g + \eta$

**3. Decoder-Signal Evolution** (Eq. 157)
$$\dot{N} = -2(NS - DM^T + g\beta s^T) - 2\lambda_A N$$

**4. Decoder-Nuisance Evolution** (Eq. 158)
$$\dot{a} = -2(Sa - \kappa s + g\rho s) - 2\lambda_A a$$

**5. Bias-Signal Evolution** (Eq. 159)
$$\dot{\beta} = -2g(Ns + \beta) - 2\lambda_b \beta$$

**6. Bias-Nuisance Evolution** (Eq. 160)
$$\dot{\rho} = -2g(a^T s - 1 + \rho) - 2\lambda_b \rho$$

**7. Classifier Evolution** (Eq. 161)
$$\dot{C} = -2(SC - gs) - 2\lambda_C C$$

**8. Encoder Autocorrelation Evolution** (Eq. 162)
$$\dot{Q} = -2(TS - G + gus^T) - 2(TS - G + gus^T)^T + 2CC^T S + 2SCC^T - 2g(Cs^T + sC^T) - 4\lambda_W Q$$

**9. Decoder Gram Matrix Evolution** (Eq. 163)
$$\dot{T} = -2(TS - G + gus^T) - 2(TS - G + gus^T)^T - 4\lambda_A T$$

**10. Decoder-Bias Inner Product Evolution** (Eq. 164)
$$\dot{u} = -2(Su - H + gms) - 2g(Ts - a + u) - 2(\lambda_A + \lambda_b)u$$

**11. Encoder-Bias Product Evolution** (Eq. 165)
$$\dot{t} = -2(TH - q + g\rho u) + 2CC^T H - 2g\rho C - 2g(B^T s - s + t) - 2(\lambda_W + \lambda_b)t$$

**12. Decoder-Encoder Correlation Evolution** (Eq. 166)
$$\dot{B} = -2(SB - S + gst^T) - 2(GT - J + gau^T) + 2GCC^T - 2gaC^T - 2(\lambda_A + \lambda_W)B$$

**13. Bias Norm Squared Evolution** (Eq. 167)
$$\dot{m} = -4g(u^T s - \rho + m) - 4\lambda_b m$$

---

## 3. Mapping to FaderNetwork Architecture

### FaderNetwork Structure (Current)
```python
class AutoEncoder(nn.Module):
    def __init__(self, n_input, n_latent):
        super().__init__()
        self.encoder = nn.Linear(n_input, n_latent)      # W ∈ ℝ^(d×n)
        self.decoder_A = nn.Linear(n_latent, n_input)    # A ∈ ℝ^(n×d)
        self.decoder_bias = nn.Parameter(torch.zeros(n_input))  # b ∈ ℝ^n

class LatentDiscriminator(nn.Module):
    def __init__(self, n_latent):
        super().__init__()
        self.classifier = nn.Linear(n_latent, 1)         # C ∈ ℝ^d
```

### Extraction Formulas for Branch B State Variables

At each training epoch, extract from trained models:

**From AutoEncoder encoder:**
```python
W = model.encoder.weight  # shape: [d, n]
```

**From AutoEncoder decoder:**
```python
A = model.decoder_A.weight  # shape: [n, d]
b = model.decoder_bias  # shape: [n]
```

**From LatentDiscriminator:**
```python
C = discriminator.classifier.weight.squeeze()  # shape: [d]
```

**Teacher components (from data):**
```python
U  # shape: [n, r] - signal subspace (orthonormal columns)
v  # shape: [n] - nuisance direction (unit norm, orthogonal to U)
Λ  # shape: [r, r] - eigenvalues of signal covariance
η  # scalar - isotropic noise level
σ_y^2 = g  # scalar - teacher output variance
```

### Computation of 13 State Variables

**Core Projected Observables:**
```python
M = W @ U                    # [d, r]
s = W @ v                    # [d]
N = U.T @ A                  # [r, d]
a = A.T @ v                  # [d]
β = U.T @ b                  # [r]
ρ = v.T @ b                  # scalar
C = C                        # [d]

Q = W @ W.T                  # [d, d]
T = A.T @ A                  # [d, d]
u = A.T @ b                  # [d]
t = W @ b                    # [d]
B = A.T @ W.T                # [d, d]
m = b.T @ b                  # scalar
```

---

## 4. Convergence Criteria for Branch B

For each epoch, verify convergence by computing **gradient norms** from the 13 equations:

```python
def compute_gradient_norms(M, s, N, a, β, ρ, C, Q, T, u, t, B, m, 
                          W, A, b, U, v, Λ, λ_W, λ_A, λ_b, λ_C, g, η):
    """
    Compute time derivatives according to Eqs. 155-167
    Returns frobenius/2-norms as convergence metrics
    """
    
    D = Λ + η * I_r
    κ = g + η
    S = M @ Λ @ M.T + g * outer(s, s) + η * Q
    G = N.T @ Λ @ M.T + g * outer(a, s) + η * B
    J = N.T @ Λ @ N + g * outer(a, a) + η * T
    H = M @ Λ @ β + g * ρ * s + η * t
    q = N.T @ Λ @ β + g * ρ * a + η * u
    
    # Compute all 13 time derivatives
    Ṁ = -2 * (T @ M @ D - N.T @ D) + 2 * outer(C, C) @ M @ D - 2 * λ_W * M
    ṡ = -2 * (κ * T @ s - κ * a + g * u) + 2 * κ * outer(C, C) @ s - 2 * g * C - 2 * λ_W * s
    Ṅ = -2 * (N @ S - D @ M.T + g * outer(β, s)) - 2 * λ_A * N
    ȧ = -2 * (S @ a - κ * s + g * ρ * s) - 2 * λ_A * a
    β̇ = -2 * g * (N @ s + β) - 2 * λ_b * β
    ρ̇ = -2 * g * (a.T @ s - 1 + ρ) - 2 * λ_b * ρ
    Ċ = -2 * (S @ C - g * s) - 2 * λ_C * C
    
    # Compute convergence metrics
    convergence_metrics = {
        'norm_M_dot': frobenius_norm(Ṁ),
        'norm_s_dot': l2_norm(ṡ),
        'norm_N_dot': frobenius_norm(Ṅ),
        'norm_a_dot': l2_norm(ȧ),
        'norm_beta_dot': l2_norm(β̇),
        'norm_rho_dot': abs(ρ̇),
        'norm_C_dot': l2_norm(Ċ),
        'max_gradient': max([...])
    }
    
    return convergence_metrics
```

**Convergence Threshold:**
- Model is converged when $\max_i |\dot{x}_i| < \epsilon$ (e.g., $\epsilon = 10^{-4}$ or $10^{-5}$)
- Track per-epoch to verify Fader converges at each training step

---

## 5. Plot Layout Specification

Based on Branch B theory, create **4-5 subplots** showing:

### Plot 1: Parameter Evolution (`subplot 1,1`)
- x-axis: Training epoch
- y-axis: Frobenius norms of key matrices
- Lines:
  - $\|M\|_F$ (encoder-signal projection)
  - $\|N\|_F$ (decoder-signal projection)
  - $\|Q\|_F$ (encoder autocorrelation)
  - $\|T\|_F$ (decoder gram matrix)
  - $\|B\|_F$ (decoder-encoder correlation)

### Plot 2: Reconstruction Error Evolution (`subplot 1,2`)
- x-axis: Training epoch
- y-axis: Loss value
- Lines:
  - Reconstruction loss: $\|x̂ - x\|^2$
  - Adversarial loss: $\|ŷ - y\|^2$

### Plot 3: Convergence Check (`subplot 1,3`)
- x-axis: Training epoch  
- y-axis: $\log_{10}(\max_i |\dot{x}_i|)$ gradient norms
- Shows convergence trajectory
- Horizontal line at convergence threshold

### Plot 4: Bias Components (`subplot 2,1`)
- x-axis: Training epoch
- y-axis: Norm values
- Lines:
  - $\|\beta\|$ (bias-signal component)
  - $|\rho|$ (bias-nuisance component)
  - $\|b\|_F$ (total bias norm, $\sqrt{m}$)

### Plot 5: Adversarial Loss Components (`subplot 2,2`)
- x-axis: Training epoch
- y-axis: Loss contribution
- Stacked or overlaid:
  - $\|s\|_F$ (encoder-nuisance projection)
  - $\|a\|_F$ (decoder-nuisance projection)
  - Trace of $\|SC - gs\|^2$ (classifier loss residual)

---

## 6. Implementation Checklist

### Step 1: Extract PDF Definitions ✓
- [x] Identify 13 state variables
- [x] Identify 13 differential equations
- [x] Identify auxiliary definitions (S, G, J, H, q)
- [x] Identify hyperparameter meanings

### Step 2: Map to FaderNetwork
- [ ] Extract W, A, b, C from trained models
- [ ] Compute U, v from teacher model
- [ ] Compute Λ, η from teacher covariance
- [ ] Implement 13-dimensional state computation

### Step 3: Verify Convergence
- [ ] Implement gradient norm computation
- [ ] Track convergence at each epoch
- [ ] Compare against theory predictions

### Step 4: Generate Plots
- [ ] Implement 5-subplot figure
- [ ] Add error bands (if multiple runs)
- [ ] Add convergence threshold lines
- [ ] Save as PNG matching PDF layout

### Step 5: Validate Theory
- [ ] Run with multiple hyperparameter settings
- [ ] Compare empirical trajectories vs. theoretical predictions
- [ ] Document any discrepancies

---

## 7. Key Constants and Hyperparameters

From `run_many_experiments.py`, ensure these are accessible:

```python
# Teacher model
n = 128              # ambient dimension
r = 5                # signal rank
λ = [0.1, 0.5, 1.0] # signal eigenvalues
σ_y^2 = 0.1         # teacher output variance
η = 0.1              # isotropic noise level

# Regularization (λ parameters)
λ_W = 0.01           # encoder regularization
λ_A = 0.01           # decoder regularization
λ_b = 0.01           # bias regularization
λ_C = 0.01           # classifier regularization

# Training
n_epochs = 100       # sufficient for convergence
batch_size = 128
learning_rate = 0.01

# Convergence threshold
convergence_threshold = 1e-5  # when to stop training
```

---

## 8. Expected Behavior

If theory matches empirical training:
1. All 13 observables should follow smooth trajectories
2. Gradient norms should decay exponentially or reach plateau
3. Parameter matrices should stabilize at fixed points
4. Reconstruction loss should follow theoretical prediction
5. Classifier should solve optimal problem exactly

### Potential Issues
- **Non-smooth evolution**: Indicates gradient flow assumption may not hold (batch effects)
- **No convergence**: May need longer training or different hyperparameters
- **Oscillations**: May indicate learning rate too high or misspecified teacher model
- **Rank deficiency**: Check if $N$, $M$, or $Q$ lose full rank unexpectedly
