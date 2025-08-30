"""
Longstaff-Schwartz Monte Carlo Benchmark
==========================================

Implements the LSM (Least-Squares Monte Carlo) method for pricing
American options. Used as a benchmark to validate PINN accuracy.

Reference:
    Longstaff & Schwartz (2001) "Valuing American Options by Simulation:
    A Simple Least-Squares Approach"
"""

import numpy as np
from typing import Tuple, Optional


def simulate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Simulate geometric Brownian motion paths.

    Parameters
    ----------
    s0 : float
        Initial spot price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to maturity.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int, optional
        Random seed.

    Returns
    -------
    paths : np.ndarray
        Simulated price paths, shape (n_steps + 1, n_paths).
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / n_steps
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = s0

    for i in range(1, n_steps + 1):
        z = np.random.standard_normal(n_paths)
        paths[i] = paths[i - 1] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        )

    return paths


def lsm_american_option(
    s0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int = 100,
    n_paths: int = 100000,
    option_type: str = "put",
    poly_degree: int = 3,
    seed: Optional[int] = 42,
) -> Tuple[float, float, np.ndarray]:
    """
    Price an American option using the Longstaff-Schwartz method.

    Parameters
    ----------
    s0 : float
        Initial spot price.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Time to maturity.
    n_steps : int
        Number of exercise dates.
    n_paths : int
        Number of Monte Carlo paths.
    option_type : str
        'put' or 'call'.
    poly_degree : int
        Degree of polynomial basis for regression.
    seed : int, optional
        Random seed.

    Returns
    -------
    price : float
        American option price.
    std_err : float
        Standard error of the estimate.
    exercise_boundary : np.ndarray
        Estimated early exercise boundary, shape (n_steps,).
    """
    dt = T / n_steps
    discount = np.exp(-r * dt)

    # Simulate paths
    paths = simulate_gbm_paths(s0, r, sigma, T, n_steps, n_paths, seed)

    # Payoff function
    if option_type == "put":
        payoff_func = lambda S: np.maximum(K - S, 0.0)
    else:
        payoff_func = lambda S: np.maximum(S - K, 0.0)

    # Initialize cash flows with payoff at maturity
    cashflows = payoff_func(paths[-1])
    exercise_time = np.full(n_paths, n_steps)
    exercise_boundary = np.full(n_steps, np.nan)

    # Backward induction
    for t in range(n_steps - 1, 0, -1):
        S_t = paths[t]
        intrinsic = payoff_func(S_t)

        # In-the-money paths
        itm = intrinsic > 0
        if np.sum(itm) < poly_degree + 1:
            continue

        S_itm = S_t[itm]

        # Discounted future cash flows for ITM paths
        # Compute continuation value
        future_cf = np.zeros(n_paths)
        for i in range(n_paths):
            if itm[i]:
                steps_ahead = exercise_time[i] - t
                future_cf[i] = cashflows[i] * discount ** steps_ahead

        Y = future_cf[itm]

        # Regression: polynomial basis
        X = np.column_stack(
            [S_itm ** k for k in range(1, poly_degree + 1)]
        )

        # Least squares fit
        try:
            coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
            continuation = X @ coeffs
        except np.linalg.LinAlgError:
            continue

        # Exercise decision: exercise if intrinsic > continuation
        exercise = intrinsic[itm] > continuation

        # Update cash flows and exercise times for exercised paths
        itm_indices = np.where(itm)[0]
        for j, idx in enumerate(itm_indices):
            if exercise[j]:
                cashflows[idx] = intrinsic[idx]
                exercise_time[idx] = t

        # Estimate exercise boundary
        if np.any(exercise):
            exercised_prices = S_itm[exercise]
            if option_type == "put":
                exercise_boundary[t] = np.max(exercised_prices)
            else:
                exercise_boundary[t] = np.min(exercised_prices)

    # Discount all cash flows to time 0
    option_values = np.zeros(n_paths)
    for i in range(n_paths):
        option_values[i] = cashflows[i] * discount ** exercise_time[i]

    # Compare with immediate exercise at t=0
    immediate = payoff_func(s0)
    option_values = np.maximum(option_values, immediate)

    price = np.mean(option_values)
    std_err = np.std(option_values) / np.sqrt(n_paths)

    return price, std_err, exercise_boundary


def compare_pinn_vs_lsm(
    pinn_pricer,
    K: float,
    r: float,
    sigma: float,
    T: float,
    s_range: Tuple[float, float] = (50, 150),
    n_points: int = 21,
    n_lsm_paths: int = 100000,
    option_type: str = "put",
) -> dict:
    """
    Compare PINN prices against LSM benchmark.

    Parameters
    ----------
    pinn_pricer : AmericanPINNPricer
        Trained PINN model.
    K, r, sigma, T : float
        Option parameters.
    s_range : tuple
        Range of spot prices to compare.
    n_points : int
        Number of comparison points.
    n_lsm_paths : int
        Number of MC paths for LSM.
    option_type : str
        'put' or 'call'.

    Returns
    -------
    results : dict
        Comparison results with prices and errors.
    """
    S_grid = np.linspace(s_range[0], s_range[1], n_points)

    pinn_prices = pinn_pricer.price(S_grid, np.zeros(n_points))
    lsm_prices = np.zeros(n_points)
    lsm_errors = np.zeros(n_points)

    print("Computing LSM benchmark prices...")
    for i, s in enumerate(S_grid):
        price, std_err, _ = lsm_american_option(
            s0=s,
            K=K,
            r=r,
            sigma=sigma,
            T=T,
            n_paths=n_lsm_paths,
            option_type=option_type,
        )
        lsm_prices[i] = price
        lsm_errors[i] = std_err
        print(
            f"  S={s:.1f}: LSM={price:.4f} (+-{std_err:.4f}), "
            f"PINN={pinn_prices[i]:.4f}"
        )

    abs_errors = np.abs(pinn_prices - lsm_prices)
    rel_errors = abs_errors / np.maximum(lsm_prices, 1e-8)

    results = {
        "S_grid": S_grid,
        "pinn_prices": pinn_prices,
        "lsm_prices": lsm_prices,
        "lsm_std_errors": lsm_errors,
        "abs_errors": abs_errors,
        "rel_errors": rel_errors,
        "mean_abs_error": np.mean(abs_errors),
        "max_abs_error": np.max(abs_errors),
        "mean_rel_error": np.mean(rel_errors[lsm_prices > 0.01]),
    }

    print(f"\nMean absolute error: {results['mean_abs_error']:.4f}")
    print(f"Max absolute error:  {results['max_abs_error']:.4f}")
    print(
        f"Mean relative error: {results['mean_rel_error']:.2%} "
        "(where LSM > 0.01)"
    )

    return results


if __name__ == "__main__":
    # Benchmark: American put option
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0

    print("=" * 60)
    print("Longstaff-Schwartz American Put Pricing")
    print("=" * 60)
    print(f"K={K}, r={r}, sigma={sigma}, T={T}")
    print()

    for s0 in [80, 90, 100, 110, 120]:
        price, std_err, boundary = lsm_american_option(
            s0=s0,
            K=K,
            r=r,
            sigma=sigma,
            T=T,
            n_steps=100,
            n_paths=200000,
            option_type="put",
        )
        print(f"  S0 = {s0:6.1f}  ->  Price = {price:.4f} +- {std_err:.4f}")
