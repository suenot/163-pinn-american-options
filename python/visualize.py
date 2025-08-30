"""
Visualization for American Option PINN
========================================

Provides:
  - Option price surface V(S, t)
  - Early exercise boundary S*(t)
  - Training loss curves
  - PINN vs LSM comparison plots
  - Greeks surfaces
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Dict, Optional
from american_pinn import AmericanPINNPricer


def plot_option_surface(
    pricer: AmericanPINNPricer,
    s_range: tuple = (50, 150),
    t_range: tuple = (0.0, 1.0),
    n_s: int = 100,
    n_t: int = 100,
    save_path: Optional[str] = None,
):
    """
    Plot the American option price surface V(S, t).

    Parameters
    ----------
    pricer : AmericanPINNPricer
        Trained PINN pricer.
    s_range : tuple
        Spot price range.
    t_range : tuple
        Time range.
    n_s, n_t : int
        Grid resolution.
    save_path : str, optional
        Path to save figure.
    """
    S_1d = np.linspace(s_range[0], s_range[1], n_s)
    t_1d = np.linspace(t_range[0], t_range[1], n_t)
    S_grid, t_grid = np.meshgrid(S_1d, t_1d)

    V = pricer.price(S_grid.flatten(), t_grid.flatten())
    V = V.reshape(n_t, n_s)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        S_grid, t_grid, V,
        cmap=cm.viridis,
        alpha=0.8,
        linewidth=0,
    )
    ax.set_xlabel("Spot Price S", fontsize=12)
    ax.set_ylabel("Time t", fontsize=12)
    ax.set_zlabel("Option Value V(S, t)", fontsize=12)
    ax.set_title(
        f"American {pricer.option_type.capitalize()} Option Surface (PINN)",
        fontsize=14,
    )
    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_exercise_boundary(
    pricer: AmericanPINNPricer,
    lsm_boundary: Optional[np.ndarray] = None,
    n_t: int = 100,
    save_path: Optional[str] = None,
):
    """
    Plot the optimal early exercise boundary S*(t).

    Parameters
    ----------
    pricer : AmericanPINNPricer
        Trained PINN pricer.
    lsm_boundary : np.ndarray, optional
        LSM benchmark boundary for comparison.
    n_t : int
        Number of time points.
    save_path : str, optional
        Path to save figure.
    """
    t_values = np.linspace(0.01, pricer.T - 0.01, n_t)
    pinn_boundary = pricer.find_exercise_boundary(t_values)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        t_values, pinn_boundary,
        "b-", linewidth=2, label="PINN Exercise Boundary",
    )

    if lsm_boundary is not None:
        t_lsm = np.linspace(0, pricer.T, len(lsm_boundary))
        valid = ~np.isnan(lsm_boundary)
        ax.plot(
            t_lsm[valid], lsm_boundary[valid],
            "r--", linewidth=2, label="LSM Exercise Boundary",
        )

    ax.axhline(y=pricer.K, color="gray", linestyle=":", label=f"Strike K={pricer.K}")

    if pricer.option_type == "put":
        ax.fill_between(
            t_values, 0, pinn_boundary,
            alpha=0.15, color="blue", label="Exercise Region",
        )
        ax.fill_between(
            t_values, pinn_boundary, pricer.s_max,
            alpha=0.15, color="green", label="Continuation Region",
        )
    else:
        ax.fill_between(
            t_values, pinn_boundary, pricer.s_max,
            alpha=0.15, color="blue", label="Exercise Region",
        )
        ax.fill_between(
            t_values, 0, pinn_boundary,
            alpha=0.15, color="green", label="Continuation Region",
        )

    ax.set_xlabel("Time t", fontsize=12)
    ax.set_ylabel("Spot Price S", fontsize=12)
    ax.set_title(
        f"American {pricer.option_type.capitalize()} Optimal Exercise Boundary",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_training_history(
    history: Dict[str, list],
    save_path: Optional[str] = None,
):
    """
    Plot training loss curves.

    Parameters
    ----------
    history : dict
        Training history from train_pinn().
    save_path : str, optional
        Path to save figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    components = [
        ("total", "Total Loss", "black"),
        ("pde", "PDE Residual", "blue"),
        ("terminal", "Terminal Condition", "green"),
        ("penalty", "Early Exercise Penalty", "red"),
    ]

    for ax, (key, label, color) in zip(axes.flatten(), components):
        values = history[key]
        ax.semilogy(values, color=color, alpha=0.7, linewidth=0.5)
        # Smoothed version
        window = min(100, len(values) // 10)
        if window > 1:
            smoothed = np.convolve(
                values, np.ones(window) / window, mode="valid"
            )
            ax.semilogy(
                range(window - 1, len(values)),
                smoothed,
                color=color, linewidth=2, label=f"{label} (smoothed)",
            )
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("PINN Training History", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_comparison(
    results: dict,
    save_path: Optional[str] = None,
):
    """
    Plot PINN vs LSM comparison.

    Parameters
    ----------
    results : dict
        Results from compare_pinn_vs_lsm().
    save_path : str, optional
        Path to save figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    S = results["S_grid"]
    pinn = results["pinn_prices"]
    lsm = results["lsm_prices"]
    lsm_err = results["lsm_std_errors"]

    # Price comparison
    ax = axes[0]
    ax.plot(S, pinn, "b-o", linewidth=2, markersize=4, label="PINN")
    ax.errorbar(
        S, lsm, yerr=2 * lsm_err,
        fmt="r--s", linewidth=2, markersize=4,
        capsize=3, label="LSM (95% CI)",
    )
    ax.set_xlabel("Spot Price S", fontsize=12)
    ax.set_ylabel("Option Price", fontsize=12)
    ax.set_title("PINN vs LSM: American Option Prices", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Error plot
    ax = axes[1]
    ax.bar(
        S, results["abs_errors"],
        width=(S[1] - S[0]) * 0.6,
        color="orange", alpha=0.7, label="Absolute Error",
    )
    ax.set_xlabel("Spot Price S", fontsize=12)
    ax.set_ylabel("Absolute Error", fontsize=12)
    ax.set_title(
        f"Pricing Error (Mean: {results['mean_abs_error']:.4f})",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_greeks(
    greeks_data: dict,
    save_path: Optional[str] = None,
):
    """
    Plot Greeks surfaces.

    Parameters
    ----------
    greeks_data : dict
        Output from greeks_surface().
    save_path : str, optional
        Path to save figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    S_grid = greeks_data["S_grid"]
    t_grid = greeks_data["t_grid"]

    surfaces = [
        ("V", "Option Value V(S, t)", cm.viridis),
        ("delta", "Delta dV/dS", cm.coolwarm),
        ("gamma", "Gamma d2V/dS2", cm.plasma),
        ("theta", "Theta dV/dt", cm.RdYlGn),
    ]

    for ax_2d, (key, title, cmap) in zip(axes.flatten(), surfaces):
        data = greeks_data[key]
        c = ax_2d.pcolormesh(
            S_grid, t_grid, data,
            cmap=cmap, shading="auto",
        )
        fig.colorbar(c, ax=ax_2d)
        ax_2d.set_xlabel("Spot Price S")
        ax_2d.set_ylabel("Time t")
        ax_2d.set_title(title, fontsize=12)

    plt.suptitle("American Option Greeks (PINN)", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_price_vs_payoff(
    pricer: AmericanPINNPricer,
    t_values: list = None,
    s_range: tuple = (50, 150),
    n_points: int = 200,
    save_path: Optional[str] = None,
):
    """
    Plot option value vs intrinsic value at different times.

    Parameters
    ----------
    pricer : AmericanPINNPricer
        Trained pricer.
    t_values : list
        Time points to plot.
    s_range : tuple
        Spot price range.
    n_points : int
        Resolution.
    save_path : str, optional
        Path to save figure.
    """
    if t_values is None:
        t_values = [0.0, 0.25, 0.5, 0.75, 0.95]

    S = np.linspace(s_range[0], s_range[1], n_points)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Intrinsic value
    if pricer.option_type == "put":
        intrinsic = np.maximum(pricer.K - S, 0)
    else:
        intrinsic = np.maximum(S - pricer.K, 0)

    ax.plot(
        S, intrinsic,
        "k--", linewidth=2, label="Intrinsic Value (payoff)",
    )

    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(t_values)))

    for t_val, color in zip(t_values, colors):
        t_arr = np.full(n_points, t_val)
        V = pricer.price(S, t_arr)
        ax.plot(
            S, V,
            color=color, linewidth=2,
            label=f"V(S, t={t_val:.2f})",
        )

    ax.set_xlabel("Spot Price S", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(
        f"American {pricer.option_type.capitalize()}: "
        "Value vs Intrinsic at Different Times",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    from train import train_pinn
    from american_pinn import create_pricer
    from lsm_benchmark import lsm_american_option, compare_pinn_vs_lsm

    # Train model
    pricer = create_pricer(
        strike=100.0,
        risk_free_rate=0.05,
        volatility=0.2,
        maturity=1.0,
        option_type="put",
    )
    history = train_pinn(pricer, n_epochs=3000)

    # Generate all visualizations
    plot_training_history(history)
    plot_option_surface(pricer)
    plot_exercise_boundary(pricer)
    plot_price_vs_payoff(pricer)

    # Compare with LSM
    results = compare_pinn_vs_lsm(
        pricer,
        K=100.0, r=0.05, sigma=0.2, T=1.0,
        n_lsm_paths=50000,
    )
    plot_comparison(results)
