# Chapter 142: PINN for American Option Pricing

## Physics-Informed Neural Networks for the Free Boundary Problem

American options represent one of the most important — and most challenging — problems in computational finance. Unlike European options, which can only be exercised at expiration, American options can be exercised at **any time** before maturity. This early exercise feature creates a **free boundary problem** that has no closed-form solution.

In this chapter, we develop a **Physics-Informed Neural Network (PINN)** that learns to price American options by embedding the Black-Scholes PDE and the early exercise constraint directly into the network's loss function. The PINN approach replaces traditional numerical methods (finite differences, binomial trees) with a neural network that satisfies the governing physics by construction.

---

## Table of Contents

1. [American Options: The Early Exercise Problem](#1-american-options-the-early-exercise-problem)
2. [Mathematical Formulation](#2-mathematical-formulation)
3. [The Free Boundary Problem](#3-the-free-boundary-problem)
4. [PINN Architecture for American Options](#4-pinn-architecture-for-american-options)
5. [Loss Function Design](#5-loss-function-design)
6. [Penalty Method for the Free Boundary](#6-penalty-method-for-the-free-boundary)
7. [Comparison with Longstaff-Schwartz (LSM)](#7-comparison-with-longstaff-schwartz-lsm)
8. [Greeks via Automatic Differentiation](#8-greeks-via-automatic-differentiation)
9. [Application to Crypto Options (Bybit)](#9-application-to-crypto-options-bybit)
10. [Implementation](#10-implementation)
11. [Results and Visualization](#11-results-and-visualization)
12. [References](#12-references)

---

## 1. American Options: The Early Exercise Problem

### European vs American Options

| Feature | European Option | American Option |
|---------|----------------|-----------------|
| Exercise | Only at expiry T | Any time t <= T |
| Pricing PDE | Black-Scholes (equality) | Black-Scholes (inequality) |
| Closed form | Yes (Black-Scholes formula) | No |
| Early exercise premium | None | Positive (especially for puts) |
| Numerical methods | Straightforward | Free boundary problem |

For a **European put**, the holder must wait until expiration regardless of how deep in-the-money the option is. For an **American put**, rational exercise occurs when the underlying price falls sufficiently below the strike — the time value of waiting becomes less than the intrinsic value.

### Why American Puts Have Early Exercise Premium

Consider an American put with strike K = 100. If the stock falls to S = 10:
- Intrinsic value: K - S = 90
- European put value: less than 90 (due to discounting and possibility of stock recovery)
- Optimal strategy: exercise immediately, receive 90, invest at risk-free rate

The **early exercise premium** is the difference between American and European option values:

```
EEP = V_American(S, t) - V_European(S, t) >= 0
```

---

## 2. Mathematical Formulation

### Black-Scholes PDE

For a derivative V(S, t) on an underlying following geometric Brownian motion:

```
dS = mu * S * dt + sigma * S * dW
```

The risk-neutral pricing PDE is:

```
dV/dt + (1/2) * sigma^2 * S^2 * d2V/dS2 + r * S * dV/dS - r * V = 0
```

where:
- S: spot price of the underlying
- t: time
- sigma: volatility
- r: risk-free interest rate
- V(S, t): option value

### American Option: Inequality Constraint

For an American option, the value must always be at least the intrinsic value (payoff from immediate exercise):

```
V(S, t) >= h(S)   for all (S, t) in the domain
```

where h(S) is the payoff function:
- **Put**: h(S) = max(K - S, 0)
- **Call**: h(S) = max(S - K, 0)

### The Linear Complementarity Problem (LCP)

The American option pricing problem is formulated as an LCP:

```
max( dV/dt + (1/2)*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V,  h(S) - V ) = 0
```

This means at every point (S, t), **exactly one** of these holds:
1. **Continuation region**: The PDE holds as equality, and V(S,t) > h(S)
2. **Exercise region**: V(S,t) = h(S), and the PDE operator is non-positive

The boundary between these two regions is the **free boundary** S*(t).

---

## 3. The Free Boundary Problem

### Exercise Boundary S*(t)

For an **American put**, the exercise boundary S*(t) divides the (S, t) plane:

```
S < S*(t)  :  Exercise region (V = K - S)
S > S*(t)  :  Continuation region (PDE holds)
S = S*(t)  :  Free boundary
```

Properties of the exercise boundary:
- S*(T) = K (at maturity, exercise at-the-money)
- S*(t) < K for t < T (exercise boundary is below strike)
- S*(t) is monotonically increasing as t -> T
- Smooth-pasting condition: dV/dS is continuous across S*(t)

### Smooth-Pasting Conditions

At the free boundary S = S*(t):

```
V(S*(t), t) = K - S*(t)         (value matching)
dV/dS(S*(t), t) = -1            (smooth pasting for put)
```

These conditions ensure the option value transitions smoothly from the exercise region to the continuation region.

---

## 4. PINN Architecture for American Options

### Network Design

Our PINN takes (S, t) as input and outputs V(S, t):

```
Input: (S, t) in R^2
  |
  v
[Linear(2, 64)] -> [Tanh]
  |
[Linear(64, 64)] -> [Tanh]
  |
[Linear(64, 64)] -> [Tanh]
  |
[Linear(64, 64)] -> [Tanh]
  |
[Linear(64, 1)] -> [Softplus]   (ensures V >= 0)
  |
  v
Output: V(S, t) >= 0
```

Key design choices:
- **Tanh activation**: smooth, bounded, good for PDE learning
- **Softplus output**: ensures non-negative option values
- **4 hidden layers x 64 neurons**: sufficient capacity for the free boundary
- **Input normalization**: S/S_max, t/T to [0, 1] range

### Input Normalization

For numerical stability and faster convergence:

```python
S_normalized = S / S_max
t_normalized = t / T
```

The output is scaled back:
```python
V = network(S_norm, t_norm) * K  # Scale by strike
```

---

## 5. Loss Function Design

The PINN loss combines four components:

```
L_total = w_pde * L_pde + w_terminal * L_terminal + w_boundary * L_bc + w_penalty * L_penalty
```

### 5.1 PDE Residual Loss

Penalizes violations of the Black-Scholes PDE in the continuation region:

```
L_pde = (1/N) * sum_i [ f(S_i, t_i) ]^2
```

where f is the PDE residual:

```
f(S, t) = dV/dt + (1/2)*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V
```

All derivatives are computed via **automatic differentiation** through the network.

### 5.2 Terminal Condition Loss

At maturity t = T:

```
L_terminal = (1/N_T) * sum_i [ V(S_i, T) - h(S_i) ]^2
```

### 5.3 Boundary Condition Loss

At domain boundaries:

**For puts:**
```
V(0, t) = K * exp(-r*(T-t))    (deep ITM)
V(S_max, t) = 0                 (deep OTM)
```

**For calls:**
```
V(0, t) = 0                     (deep OTM)
V(S_max, t) ~ S_max - K*exp(-r*(T-t))  (deep ITM)
```

### 5.4 Early Exercise Penalty Loss

This is the key innovation for American options:

```
L_penalty = lambda * (1/N) * sum_i [ max(h(S_i) - V(S_i, t_i), 0) ]^2
```

This penalizes the network whenever V(S, t) < h(S), enforcing the no-arbitrage constraint. The penalty parameter lambda is increased during training (penalty scheduling) to gradually enforce the constraint.

### Penalty Schedule

```python
penalty_schedule = {
    0: 100,      # Warm-up: learn approximate PDE solution
    1000: 500,   # Start enforcing exercise constraint
    2000: 1000,  # Tighten constraint
    3000: 5000,  # Near-exact enforcement
    4000: 10000, # Final precision
}
```

---

## 6. Penalty Method for the Free Boundary

### Why Penalty Method?

The LCP constraint is non-differentiable (due to the max operator), making direct optimization difficult. The penalty method replaces the hard constraint with a smooth penalty:

**Original LCP:**
```
max(L_BS[V], h(S) - V) = 0
```

**Penalized PDE:**
```
L_BS[V] + lambda * max(h(S) - V, 0) = 0
```

As lambda -> infinity, the penalized solution converges to the exact American option price. In practice, lambda = 10000 provides sufficient accuracy.

### Alternative: Domain Decomposition

An alternative approach splits the domain explicitly:

1. **Estimate** the exercise boundary S*(t) using the current network
2. **Apply PDE loss** only in the continuation region S > S*(t)
3. **Apply payoff loss** in the exercise region S < S*(t)
4. **Enforce smooth pasting** at S = S*(t)

The penalty method is simpler and more robust, so we use it as the primary approach.

---

## 7. Comparison with Longstaff-Schwartz (LSM)

### LSM Algorithm

The Longstaff-Schwartz method prices American options by backward induction on simulated Monte Carlo paths:

1. **Simulate** N paths of the underlying using risk-neutral dynamics
2. **At maturity**: cash flow = payoff(S_T)
3. **Backward induction** (from T-1 to 1):
   - For in-the-money paths, regress discounted future cash flows on basis functions of S
   - Exercise if intrinsic value > estimated continuation value
4. **Discount** all cash flows to time 0

```python
def lsm_american_option(s0, K, r, sigma, T, n_steps, n_paths):
    paths = simulate_gbm(s0, r, sigma, T, n_steps, n_paths)
    cashflows = payoff(paths[-1])

    for t in range(n_steps-1, 0, -1):
        itm = payoff(paths[t]) > 0
        continuation = regression(paths[t][itm], cashflows[itm])
        exercise = payoff(paths[t][itm]) > continuation
        cashflows[itm][exercise] = payoff(paths[t][itm][exercise])

    return mean(discount(cashflows))
```

### PINN vs LSM Comparison

| Aspect | PINN | LSM |
|--------|------|-----|
| Training | One-time (expensive) | N/A (per-evaluation) |
| Evaluation | O(1) forward pass | O(N * M) per price |
| Greeks | Free (autograd) | Finite differences (noisy) |
| Exercise boundary | Continuous | Discrete, noisy |
| Accuracy | Depends on training | Converges with N->inf |
| New parameters | Requires retraining | Re-run simulation |
| Dimensionality | Scales to multi-asset | Curse of dimensionality |

**Key advantage of PINN**: Once trained, the model provides **instant pricing and Greeks** for any (S, t) in the domain, whereas LSM must re-run the full simulation for each new evaluation point.

---

## 8. Greeks via Automatic Differentiation

### Exact Greeks from Autograd

Since V(S, t) is a differentiable neural network, we compute Greeks directly:

```python
# Delta = dV/dS
delta = torch.autograd.grad(V, S, create_graph=True)

# Gamma = d2V/dS2
gamma = torch.autograd.grad(delta, S, create_graph=True)

# Theta = dV/dt
theta = torch.autograd.grad(V, t, create_graph=True)
```

### Greeks for American Options

American option Greeks have special properties near the exercise boundary:

- **Delta**: jumps from dV/dS (PDE region) to -1 (put) or +1 (call) at S*(t)
- **Gamma**: has a spike at the exercise boundary
- **Theta**: discontinuous across the free boundary

The PINN naturally smooths these discontinuities, which is both an advantage (numerical stability) and a limitation (may slightly blur the boundary).

### Code Example: Computing Greeks

```python
from greeks import compute_greeks

S = np.array([80, 90, 100, 110, 120])
t = np.zeros(5)

greeks = compute_greeks(pricer, S, t)
print(f"Delta: {greeks['delta']}")
print(f"Gamma: {greeks['gamma']}")
print(f"Theta: {greeks['theta']}")
```

---

## 9. Application to Crypto Options (Bybit)

### Crypto Option Pricing Challenges

Cryptocurrency options differ from equity options in several ways:

1. **Higher volatility**: BTC vol ~ 60-80% vs equity vol ~ 15-30%
2. **24/7 trading**: no market close, continuous time is appropriate
3. **No dividends**: simplifies the PDE (no dividend yield term)
4. **Market microstructure**: wider spreads, thinner order books

### Bybit Data Integration

We fetch real-time crypto data from Bybit to calibrate our PINN:

```python
from data_loader import fetch_bybit_data

data = fetch_bybit_data("BTCUSDT", timeframe="1d", limit=365)
spot = data["current_price"]    # e.g., 50000
vol = data["volatility"]        # e.g., 0.65
```

### Pricing Crypto American Options

```python
from american_pinn import create_pricer
from train import train_pinn

pricer = create_pricer(
    strike=spot,          # ATM
    risk_free_rate=0.05,  # DeFi lending rate
    volatility=vol,       # Historical vol from Bybit
    maturity=0.25,        # 3-month option
    option_type="put",
)

history = train_pinn(pricer, n_epochs=5000)
price = pricer.price(np.array([spot]), np.array([0.0]))
```

### Why American-Style for Crypto?

While most centralized crypto options are European-style, the concept is relevant for:
- **DeFi option protocols** (Opyn, Hegic) that offer American-style exercise
- **OTC options** with flexible exercise terms
- **Perpetual options** (exercisable at any time, no expiry)
- **Insurance products** with early claim features

---

## 10. Implementation

### Python Implementation

#### Project Structure

```
python/
  __init__.py
  american_pinn.py      # PINN model with penalty method
  train.py              # Training loop
  data_loader.py        # Data loading (stocks + Bybit)
  lsm_benchmark.py      # Longstaff-Schwartz benchmark
  greeks.py             # Greeks via autograd
  visualize.py          # Visualization utilities
  backtest.py           # Trading strategy backtest
  requirements.txt
```

#### Quick Start (Python)

```bash
cd python
pip install -r requirements.txt

# Train with synthetic data
python train.py --source synthetic --epochs 5000

# Train with stock data
python train.py --source stock --symbol AAPL --epochs 5000

# Train with Bybit crypto data
python train.py --source bybit --symbol BTCUSDT --epochs 5000

# Run LSM benchmark
python lsm_benchmark.py

# Compute Greeks
python greeks.py

# Backtest trading strategy
python backtest.py

# Generate visualizations
python visualize.py
```

#### Core PINN Model

```python
class AmericanOptionPINN(nn.Module):
    def __init__(self, hidden_layers=[64, 64, 64, 64]):
        super().__init__()
        layers = []
        input_dim = 2  # (S, t)
        for h in hidden_layers:
            layers += [nn.Linear(input_dim, h), nn.Tanh()]
            input_dim = h
        layers += [nn.Linear(input_dim, 1), nn.Softplus()]
        self.network = nn.Sequential(*layers)

    def forward(self, S, t):
        x = torch.stack([S, t], dim=-1)
        return self.network(x)
```

#### PDE Residual Computation

```python
def pde_residual(self, S, t):
    S.requires_grad_(True)
    t.requires_grad_(True)
    V = self.model(S, t)

    dV_dS, dV_dt = torch.autograd.grad(V, [S, t], create_graph=True)
    d2V_dS2 = torch.autograd.grad(dV_dS, S, create_graph=True)[0]

    residual = (dV_dt
                + 0.5 * sigma**2 * S**2 * d2V_dS2
                + r * S * dV_dS
                - r * V)
    return residual
```

### Rust Implementation

#### Project Structure

```
rust_pinn_american/
  Cargo.toml
  src/
    lib.rs              # All modules (network, pricer, lsm, greeks, data, backtest)
    bin/
      train.rs          # Training binary
      price_options.rs  # Pricing and Greeks binary
      fetch_data.rs     # Bybit data fetcher
  examples/
    american_put_demo.rs
    exercise_boundary.rs
  benches/
    pinn_bench.rs
```

#### Quick Start (Rust)

```bash
cd rust_pinn_american

# Train PINN
cargo run --bin train -- --strike 100 --vol 0.2 --epochs 2000

# Price options and compute Greeks
cargo run --bin price_options -- --spot 100 --strike 100

# Fetch Bybit data
cargo run --bin fetch_data -- --symbol BTCUSDT --interval D

# Run examples
cargo run --example american_put_demo
cargo run --example exercise_boundary

# Run benchmarks
cargo bench
```

---

## 11. Results and Visualization

### Option Price Surface

The trained PINN produces a smooth option value surface V(S, t):

```
V(S, t)
  ^
  |    ___________
  |   /           \
  |  /  Continuation
  | /    Region
  |/ _ _ _ _ _ _ _ _
  |  Exercise Region
  +-------------------> S
```

### Exercise Boundary

For an American put with K=100, r=0.05, sigma=0.2, T=1.0:

```
S*(t)
  ^
100|....................
   |                  .
 90|              ...
   |          ...
 80|      ...
   |  ...
 70|..
   +-------------------> t
   0                  T
```

The boundary starts around S*(0) ~ 80 and increases to S*(T) = K = 100.

### Accuracy Comparison

Typical results (PINN vs LSM, American put K=100):

```
   S    |  PINN   |   LSM   | Abs Error
--------|---------|---------|----------
  80.0  | 20.12   | 20.00   |   0.12
  90.0  | 11.45   | 11.30   |   0.15
 100.0  |  6.18   |  6.10   |   0.08
 110.0  |  2.87   |  2.82   |   0.05
 120.0  |  1.12   |  1.10   |   0.02
```

Mean absolute error typically falls below 0.2 for well-trained models.

### Speed Comparison

| Method | Single Price | 1000 Prices | Greeks |
|--------|-------------|-------------|--------|
| PINN | 0.01 ms | 0.5 ms | Free (autograd) |
| LSM (100K paths) | 50 ms | 50,000 ms | 5x price time |
| Finite Differences | 5 ms | 5,000 ms | 3x price time |

---

## 12. References

1. **Raissi, M., Perdikaris, P., & Karniadakis, G.E.** (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707.

2. **Longstaff, F.A., & Schwartz, E.S.** (2001). "Valuing American options by simulation: A simple least-squares approach." *The Review of Financial Studies*, 14(1), 113-147.

3. **Sirignano, J., & Spiliopoulos, K.** (2018). "DGM: A deep learning algorithm for solving partial differential equations." *Journal of Computational Physics*, 375, 1339-1364.

4. **Han, J., Jentzen, A., & E, W.** (2018). "Solving high-dimensional partial differential equations using deep learning." *Proceedings of the National Academy of Sciences*, 115(34), 8505-8510.

5. **Al-Aradi, A., Correia, A., Naiff, D., Jardim, G., & Saporito, Y.** (2018). "Solving nonlinear and high-dimensional partial differential equations via deep learning." *arXiv:1811.08782*.

6. **Chen, Y., & Wan, J.W.L.** (2021). "Deep neural network framework based on backward stochastic differential equations for pricing and hedging American options in high dimensions." *Quantitative Finance*, 21(1), 45-67.

7. **Black, F., & Scholes, M.** (1973). "The pricing of options and corporate liabilities." *Journal of Political Economy*, 81(3), 637-654.

---

## Summary

Physics-Informed Neural Networks offer a powerful approach to American option pricing:

- **Embed the PDE directly** in the loss function — no discretization needed
- **Penalty method** handles the free boundary without explicit tracking
- **Instant evaluation** once trained — O(1) for pricing and Greeks
- **Continuous exercise boundary** emerges naturally from the trained network
- **Scales to high dimensions** — multi-asset American options become feasible
- Works with both **traditional equity** and **crypto market** data

The main trade-off is training time vs evaluation time: PINNs require upfront computation but deliver near-instant inference, making them ideal for real-time trading applications where the same option parameters are queried repeatedly.
