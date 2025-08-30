"""
Training Script for American Option PINN
==========================================

Trains the PINN model with:
  - Adaptive penalty scheduling (increasing lambda during training)
  - Learning rate scheduling
  - Collocation point resampling
  - Loss monitoring and early stopping
"""

import torch
import numpy as np
import time
from typing import Optional, Dict, Tuple
from tqdm import tqdm

from american_pinn import AmericanPINNPricer, create_pricer
from data_loader import (
    generate_collocation_points,
    prepare_training_data,
    fetch_stock_data,
    fetch_bybit_data,
)


def train_pinn(
    pricer: AmericanPINNPricer,
    n_epochs: int = 5000,
    learning_rate: float = 1e-3,
    n_interior: int = 10000,
    n_boundary: int = 500,
    n_terminal: int = 500,
    resample_every: int = 1000,
    penalty_schedule: Optional[Dict[int, float]] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    verbose: bool = True,
    save_path: Optional[str] = None,
) -> Dict[str, list]:
    """
    Train PINN for American option pricing.

    Parameters
    ----------
    pricer : AmericanPINNPricer
        The PINN pricer to train.
    n_epochs : int
        Number of training epochs.
    learning_rate : float
        Initial learning rate.
    n_interior : int
        Number of interior collocation points.
    n_boundary : int
        Number of boundary points.
    n_terminal : int
        Number of terminal points.
    resample_every : int
        Resample collocation points every N epochs.
    penalty_schedule : dict, optional
        {epoch: penalty_lambda} for adaptive penalty.
    loss_weights : dict, optional
        Loss component weights.
    verbose : bool
        Print training progress.
    save_path : str, optional
        Path to save model after training.

    Returns
    -------
    history : dict
        Training loss history.
    """
    if penalty_schedule is None:
        penalty_schedule = {
            0: 100.0,
            1000: 500.0,
            2000: 1000.0,
            3000: 5000.0,
            4000: 10000.0,
        }

    if loss_weights is None:
        loss_weights = {
            "pde": 1.0,
            "terminal": 10.0,
            "boundary": 5.0,
            "penalty": 1.0,
        }

    optimizer = torch.optim.Adam(
        pricer.model.parameters(), lr=learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )

    history = {
        "total": [],
        "pde": [],
        "terminal": [],
        "boundary": [],
        "penalty": [],
    }

    # Generate initial collocation points
    data = prepare_training_data(
        strike=pricer.K,
        volatility=pricer.sigma,
        maturity=pricer.T,
        n_interior=n_interior,
        n_boundary=n_boundary,
        n_terminal=n_terminal,
        device=str(pricer.device),
    )

    start_time = time.time()

    pbar = tqdm(range(n_epochs), disable=not verbose, desc="Training PINN")
    for epoch in pbar:
        # Update penalty parameter
        if epoch in penalty_schedule:
            pricer.penalty_lambda = penalty_schedule[epoch]
            if verbose:
                pbar.write(
                    f"  [Epoch {epoch}] Penalty lambda -> "
                    f"{pricer.penalty_lambda:.0f}"
                )

        # Resample collocation points
        if epoch > 0 and epoch % resample_every == 0:
            data = prepare_training_data(
                strike=pricer.K,
                volatility=pricer.sigma,
                maturity=pricer.T,
                n_interior=n_interior,
                n_boundary=n_boundary,
                n_terminal=n_terminal,
                device=str(pricer.device),
            )
            if verbose:
                pbar.write(f"  [Epoch {epoch}] Resampled collocation points")

        pricer.model.train()
        optimizer.zero_grad()

        # Compute loss
        total_loss, loss_dict = pricer.compute_loss(
            S_interior=data.S_interior,
            t_interior=data.t_interior,
            S_boundary=data.S_boundary,
            t_boundary=data.t_boundary,
            S_terminal=data.S_terminal,
            weights=loss_weights,
        )

        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(pricer.model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Record history
        for key in history:
            history[key].append(loss_dict[key])

        if verbose and epoch % 500 == 0:
            elapsed = time.time() - start_time
            pbar.set_postfix(
                loss=f"{loss_dict['total']:.6f}",
                pde=f"{loss_dict['pde']:.6f}",
                penalty=f"{loss_dict['penalty']:.6f}",
                time=f"{elapsed:.1f}s",
            )

    elapsed = time.time() - start_time
    if verbose:
        print(f"\nTraining complete in {elapsed:.1f}s")
        print(f"Final loss: {history['total'][-1]:.6f}")

    if save_path:
        pricer.save(save_path)

    return history


def train_with_market_data(
    source: str = "stock",
    symbol: str = "AAPL",
    strike_pct: float = 1.0,
    option_type: str = "put",
    maturity: float = 0.25,
    n_epochs: int = 5000,
    device: str = "cpu",
) -> Tuple[AmericanPINNPricer, Dict[str, list]]:
    """
    Train PINN using real market data for parameter calibration.

    Parameters
    ----------
    source : str
        'stock' for equity data, 'bybit' for crypto data.
    symbol : str
        Ticker / trading pair.
    strike_pct : float
        Strike as fraction of current price (1.0 = ATM).
    option_type : str
        'put' or 'call'.
    maturity : float
        Time to maturity in years.
    n_epochs : int
        Training epochs.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    pricer : AmericanPINNPricer
        Trained pricer.
    history : dict
        Training history.
    """
    # Fetch market data
    if source == "stock":
        market = fetch_stock_data(symbol)
        risk_free_rate = 0.05  # approximate
    elif source == "bybit":
        market = fetch_bybit_data(symbol)
        risk_free_rate = 0.05  # DeFi lending rate approximation
    else:
        raise ValueError(f"Unknown source: {source}")

    spot = market["current_price"]
    vol = market["volatility"]
    strike = spot * strike_pct

    print(f"Market data for {market['symbol']}:")
    print(f"  Current price: {spot:.2f}")
    print(f"  Historical vol: {vol:.4f}")
    print(f"  Strike ({strike_pct:.0%}): {strike:.2f}")
    print(f"  Option type: {option_type}")
    print(f"  Maturity: {maturity:.2f}y")

    pricer = create_pricer(
        strike=strike,
        risk_free_rate=risk_free_rate,
        volatility=vol,
        maturity=maturity,
        option_type=option_type,
        device=device,
    )

    history = train_pinn(
        pricer,
        n_epochs=n_epochs,
        learning_rate=1e-3,
        n_interior=10000,
        verbose=True,
    )

    return pricer, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train PINN for American option pricing"
    )
    parser.add_argument(
        "--source", type=str, default="stock",
        choices=["stock", "bybit", "synthetic"],
        help="Data source",
    )
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--option-type", type=str, default="put",
                        choices=["put", "call"])
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--volatility", type=float, default=0.2)
    parser.add_argument("--maturity", type=float, default=1.0)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    if args.source == "synthetic":
        print("=" * 60)
        print("Training PINN with synthetic parameters")
        print("=" * 60)
        pricer = create_pricer(
            strike=args.strike,
            risk_free_rate=args.rate,
            volatility=args.volatility,
            maturity=args.maturity,
            option_type=args.option_type,
            device=args.device,
        )
        history = train_pinn(
            pricer,
            n_epochs=args.epochs,
            save_path=args.save,
        )
    else:
        print("=" * 60)
        print(f"Training PINN with {args.source} data ({args.symbol})")
        print("=" * 60)
        pricer, history = train_with_market_data(
            source=args.source,
            symbol=args.symbol,
            option_type=args.option_type,
            maturity=args.maturity,
            n_epochs=args.epochs,
            device=args.device,
        )

    # Price a grid for demonstration
    S_grid = np.linspace(50, 150, 11)
    t_grid = np.full_like(S_grid, 0.0)  # price at t=0
    prices = pricer.price(S_grid, t_grid)

    print("\n" + "=" * 60)
    print("Option Prices at t=0:")
    print("=" * 60)
    for s, p in zip(S_grid, prices):
        print(f"  S = {s:8.2f}  ->  V = {p:8.4f}")
