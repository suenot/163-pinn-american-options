"""
Backtest for American Option Trading Strategy
===============================================

Implements a trading strategy based on PINN-detected mispricings:
  1. Compute theoretical American option price using PINN
  2. Compare with market price (or LSM benchmark)
  3. Trade when mispricing exceeds threshold
  4. Delta-hedge the position

Supports both stock and crypto (Bybit) scenarios.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from american_pinn import AmericanPINNPricer
from lsm_benchmark import lsm_american_option


@dataclass
class Trade:
    """Represents a single option trade."""
    entry_time: int
    exit_time: int = -1
    entry_spot: float = 0.0
    exit_spot: float = 0.0
    entry_price: float = 0.0  # option price at entry
    exit_price: float = 0.0   # option price at exit
    pinn_value: float = 0.0   # PINN theoretical value
    market_value: float = 0.0 # market price
    direction: str = "long"   # 'long' or 'short'
    pnl: float = 0.0
    delta_hedge_pnl: float = 0.0


@dataclass
class BacktestResult:
    """Results from backtest."""
    total_pnl: float = 0.0
    n_trades: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    daily_pnl: np.ndarray = field(default_factory=lambda: np.array([]))


def simulate_market_prices(
    pinn_prices: np.ndarray,
    noise_level: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate market prices with noise around theoretical values.

    In a real scenario, market prices come from the exchange order book.
    Here we simulate mispricings by adding noise to PINN values.

    Parameters
    ----------
    pinn_prices : np.ndarray
        Theoretical PINN prices.
    noise_level : float
        Standard deviation of noise as fraction of price.
    seed : int
        Random seed.

    Returns
    -------
    market_prices : np.ndarray
        Simulated market prices.
    """
    np.random.seed(seed)
    noise = np.random.normal(0, noise_level, size=pinn_prices.shape)
    market_prices = pinn_prices * (1.0 + noise)
    return np.maximum(market_prices, 0.0)


def run_backtest(
    pricer: AmericanPINNPricer,
    spot_prices: np.ndarray,
    entry_threshold: float = 0.05,
    exit_threshold: float = 0.01,
    max_holding_days: int = 20,
    position_size: float = 1.0,
    transaction_cost: float = 0.001,
    delta_hedge: bool = True,
) -> BacktestResult:
    """
    Run backtest of mispricing-based option trading strategy.

    Strategy:
      - At each time step, compute PINN theoretical price
      - Generate a "market price" (with noise for simulation)
      - If PINN_price > market_price by threshold -> buy option
      - If PINN_price < market_price by threshold -> sell option
      - Exit when mispricing narrows or max holding period reached
      - Optionally delta-hedge the underlying

    Parameters
    ----------
    pricer : AmericanPINNPricer
        Trained PINN model.
    spot_prices : np.ndarray
        Historical spot price series.
    entry_threshold : float
        Minimum mispricing ratio to enter trade.
    exit_threshold : float
        Exit when mispricing narrows to this level.
    max_holding_days : int
        Maximum holding period.
    position_size : float
        Number of option contracts.
    transaction_cost : float
        Transaction cost as fraction.
    delta_hedge : bool
        Whether to delta-hedge positions.

    Returns
    -------
    result : BacktestResult
        Backtest results.
    """
    n_days = len(spot_prices)
    trades = []
    equity_curve = np.zeros(n_days)
    daily_pnl = np.zeros(n_days)
    current_equity = 0.0

    # Generate time values (decreasing as we approach maturity)
    # Assume we start at t=0 with T time to expiry
    T = pricer.T
    time_values = np.linspace(0.0, T * 0.9, n_days)

    # Compute PINN prices along the path
    pinn_prices = pricer.price(spot_prices, time_values)
    market_prices = simulate_market_prices(pinn_prices, noise_level=0.03)

    # Compute deltas for hedging
    deltas = np.zeros(n_days)
    if delta_hedge:
        import torch
        for i in range(n_days):
            S_t = torch.tensor(
                [spot_prices[i]], dtype=torch.float32,
                device=pricer.device, requires_grad=True,
            )
            t_t = torch.tensor(
                [time_values[i]], dtype=torch.float32,
                device=pricer.device,
            )
            V = pricer.model(S_t, t_t)
            V.backward()
            deltas[i] = S_t.grad.item()

    active_trade = None

    for i in range(n_days):
        pinn_v = pinn_prices[i]
        mkt_v = market_prices[i]

        if pinn_v < 1e-6 and mkt_v < 1e-6:
            equity_curve[i] = current_equity
            continue

        mispricing = (pinn_v - mkt_v) / max(pinn_v, 1e-8)

        if active_trade is None:
            # Check for entry
            if abs(mispricing) > entry_threshold:
                active_trade = Trade(
                    entry_time=i,
                    entry_spot=spot_prices[i],
                    entry_price=mkt_v,
                    pinn_value=pinn_v,
                    market_value=mkt_v,
                    direction="long" if mispricing > 0 else "short",
                )
        else:
            # Check for exit
            holding_time = i - active_trade.entry_time
            current_mispricing = abs(mispricing)

            should_exit = (
                current_mispricing < exit_threshold
                or holding_time >= max_holding_days
                or i == n_days - 1
            )

            if should_exit:
                active_trade.exit_time = i
                active_trade.exit_spot = spot_prices[i]
                active_trade.exit_price = mkt_v

                if active_trade.direction == "long":
                    raw_pnl = (mkt_v - active_trade.entry_price) * position_size
                else:
                    raw_pnl = (active_trade.entry_price - mkt_v) * position_size

                # Transaction costs
                cost = transaction_cost * (
                    active_trade.entry_price + mkt_v
                ) * position_size
                raw_pnl -= cost

                # Delta hedge P&L
                hedge_pnl = 0.0
                if delta_hedge and active_trade.entry_time < len(deltas):
                    # Approximate hedge P&L
                    delta_entry = deltas[active_trade.entry_time]
                    spot_change = spot_prices[i] - active_trade.entry_spot
                    if active_trade.direction == "long":
                        hedge_pnl = -delta_entry * spot_change * position_size
                    else:
                        hedge_pnl = delta_entry * spot_change * position_size

                active_trade.pnl = raw_pnl
                active_trade.delta_hedge_pnl = hedge_pnl
                trades.append(active_trade)

                current_equity += raw_pnl + hedge_pnl
                daily_pnl[i] = raw_pnl + hedge_pnl
                active_trade = None

        equity_curve[i] = current_equity

    # Compute statistics
    result = BacktestResult()
    result.trades = trades
    result.n_trades = len(trades)
    result.equity_curve = equity_curve
    result.daily_pnl = daily_pnl

    if trades:
        pnls = np.array([t.pnl + t.delta_hedge_pnl for t in trades])
        result.total_pnl = np.sum(pnls)
        result.avg_pnl = np.mean(pnls)
        result.win_rate = np.mean(pnls > 0) if len(pnls) > 0 else 0.0

        # Max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        result.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        # Sharpe ratio (annualized)
        if np.std(pnls) > 0:
            result.sharpe_ratio = (
                np.mean(pnls) / np.std(pnls) * np.sqrt(252)
            )

    return result


def print_backtest_report(result: BacktestResult, title: str = "Backtest"):
    """Print formatted backtest report."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(f"  Total P&L:     {result.total_pnl:>12.4f}")
    print(f"  Num Trades:    {result.n_trades:>12d}")
    print(f"  Win Rate:      {result.win_rate:>12.2%}")
    print(f"  Avg P&L:       {result.avg_pnl:>12.4f}")
    print(f"  Max Drawdown:  {result.max_drawdown:>12.4f}")
    print(f"  Sharpe Ratio:  {result.sharpe_ratio:>12.4f}")
    print("=" * 60)

    if result.trades:
        print("\n  Last 5 trades:")
        for t in result.trades[-5:]:
            print(
                f"    [{t.entry_time:4d}->{t.exit_time:4d}] "
                f"{t.direction:5s} "
                f"S: {t.entry_spot:.2f}->{t.exit_spot:.2f} "
                f"PnL: {t.pnl + t.delta_hedge_pnl:+.4f}"
            )


if __name__ == "__main__":
    from train import train_pinn
    from american_pinn import create_pricer
    from data_loader import fetch_stock_data, fetch_bybit_data

    # ------ Stock Backtest ------
    print("\n" + "#" * 60)
    print("# Stock Market Backtest (American Put)")
    print("#" * 60)

    stock_data = fetch_stock_data("AAPL")
    spot = stock_data["current_price"]
    vol = stock_data["volatility"]

    pricer_stock = create_pricer(
        strike=spot,
        risk_free_rate=0.05,
        volatility=vol,
        maturity=0.25,
        option_type="put",
    )
    train_pinn(pricer_stock, n_epochs=3000, verbose=True)

    result_stock = run_backtest(
        pricer_stock,
        spot_prices=stock_data["prices"],
        entry_threshold=0.05,
        delta_hedge=True,
    )
    print_backtest_report(result_stock, "Stock (AAPL) American Put Backtest")

    # ------ Crypto Backtest ------
    print("\n" + "#" * 60)
    print("# Crypto Market Backtest (American Put on BTCUSDT)")
    print("#" * 60)

    crypto_data = fetch_bybit_data("BTCUSDT")
    spot_c = crypto_data["current_price"]
    vol_c = crypto_data["volatility"]

    pricer_crypto = create_pricer(
        strike=spot_c,
        risk_free_rate=0.05,
        volatility=vol_c,
        maturity=0.25,
        option_type="put",
    )
    train_pinn(pricer_crypto, n_epochs=3000, verbose=True)

    result_crypto = run_backtest(
        pricer_crypto,
        spot_prices=crypto_data["prices"],
        entry_threshold=0.05,
        delta_hedge=True,
    )
    print_backtest_report(
        result_crypto, "Crypto (BTCUSDT) American Put Backtest"
    )
