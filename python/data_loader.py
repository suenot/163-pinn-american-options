"""
Data Loader for American Option PINN
=====================================

Provides:
  - Synthetic collocation points for PINN training (interior, boundary, terminal)
  - Real stock data via yfinance
  - Crypto data from Bybit exchange via CCXT
  - Implied volatility estimation from market data
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class CollocationData:
    """Collocation points for PINN training."""
    S_interior: torch.Tensor
    t_interior: torch.Tensor
    S_boundary: torch.Tensor
    t_boundary: torch.Tensor
    S_terminal: torch.Tensor


def generate_collocation_points(
    s_min: float = 0.0,
    s_max: float = 300.0,
    t_max: float = 1.0,
    n_interior: int = 10000,
    n_boundary: int = 500,
    n_terminal: int = 500,
    device: str = "cpu",
    use_sobol: bool = False,
) -> CollocationData:
    """
    Generate collocation points for PINN training.

    Uses Latin Hypercube or Sobol sampling for better coverage
    of the (S, t) domain.

    Parameters
    ----------
    s_min : float
        Minimum spot price.
    s_max : float
        Maximum spot price.
    t_max : float
        Maximum time (maturity).
    n_interior : int
        Number of interior collocation points.
    n_boundary : int
        Number of boundary condition points.
    n_terminal : int
        Number of terminal condition points.
    device : str
        Torch device.
    use_sobol : bool
        Use Sobol quasi-random sequence for better coverage.

    Returns
    -------
    data : CollocationData
        Collocation points on the specified device.
    """
    dev = torch.device(device)

    if use_sobol:
        try:
            sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
            interior_samples = sobol.draw(n_interior)
            S_int = s_min + (s_max - s_min) * interior_samples[:, 0]
            t_int = t_max * interior_samples[:, 1]
        except Exception:
            # Fallback to random
            S_int = s_min + (s_max - s_min) * torch.rand(n_interior)
            t_int = t_max * torch.rand(n_interior)
    else:
        S_int = s_min + (s_max - s_min) * torch.rand(n_interior)
        t_int = t_max * torch.rand(n_interior)

    # Boundary points (uniform in time)
    t_bnd = t_max * torch.rand(n_boundary)

    # Terminal condition points (uniform in S at t = T)
    S_term = s_min + (s_max - s_min) * torch.rand(n_terminal)

    return CollocationData(
        S_interior=S_int.to(dev),
        t_interior=t_int.to(dev),
        S_boundary=t_bnd.to(dev),
        t_boundary=t_bnd.to(dev),
        S_terminal=S_term.to(dev),
    )


def fetch_stock_data(
    symbol: str = "AAPL",
    period: str = "1y",
) -> Dict[str, np.ndarray]:
    """
    Fetch stock data using yfinance.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol.
    period : str
        Data period (e.g., '1y', '2y', '6mo').

    Returns
    -------
    data : dict
        Dictionary with 'prices', 'returns', 'volatility', 'dates'.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Using synthetic data.")
        return _generate_synthetic_stock_data()

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)

        if hist.empty:
            print(f"No data for {symbol}. Using synthetic data.")
            return _generate_synthetic_stock_data()

        prices = hist["Close"].values
        returns = np.diff(np.log(prices))
        vol = np.std(returns) * np.sqrt(252)

        return {
            "symbol": symbol,
            "prices": prices,
            "returns": returns,
            "volatility": vol,
            "dates": hist.index.values,
            "current_price": prices[-1],
        }
    except Exception as e:
        print(f"Error fetching {symbol}: {e}. Using synthetic data.")
        return _generate_synthetic_stock_data()


def fetch_bybit_data(
    symbol: str = "BTCUSDT",
    timeframe: str = "1d",
    limit: int = 365,
) -> Dict[str, np.ndarray]:
    """
    Fetch cryptocurrency data from Bybit via CCXT.

    Parameters
    ----------
    symbol : str
        Trading pair (e.g., 'BTCUSDT', 'ETHUSDT').
    timeframe : str
        Candle timeframe ('1d', '4h', '1h').
    limit : int
        Number of candles to fetch.

    Returns
    -------
    data : dict
        Dictionary with 'prices', 'returns', 'volatility', etc.
    """
    try:
        import ccxt
    except ImportError:
        print("ccxt not installed. Using synthetic crypto data.")
        return _generate_synthetic_crypto_data(symbol)

    try:
        exchange = ccxt.bybit({"enableRateLimit": True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        if not ohlcv:
            print(f"No data from Bybit for {symbol}. Using synthetic data.")
            return _generate_synthetic_crypto_data(symbol)

        data = np.array(ohlcv)
        timestamps = data[:, 0]
        opens = data[:, 1]
        highs = data[:, 2]
        lows = data[:, 3]
        closes = data[:, 4]
        volumes = data[:, 5]

        returns = np.diff(np.log(closes))
        vol = np.std(returns) * np.sqrt(365)  # annualized for crypto

        return {
            "symbol": symbol,
            "timestamps": timestamps,
            "opens": opens,
            "highs": highs,
            "lows": lows,
            "closes": closes,
            "volumes": volumes,
            "prices": closes,
            "returns": returns,
            "volatility": vol,
            "current_price": closes[-1],
        }
    except Exception as e:
        print(f"Error fetching Bybit data: {e}. Using synthetic data.")
        return _generate_synthetic_crypto_data(symbol)


def _generate_synthetic_stock_data(
    s0: float = 150.0,
    mu: float = 0.08,
    sigma: float = 0.2,
    n_days: int = 252,
) -> Dict[str, np.ndarray]:
    """Generate synthetic stock price data using GBM."""
    dt = 1.0 / 252
    prices = np.zeros(n_days)
    prices[0] = s0

    np.random.seed(42)
    for i in range(1, n_days):
        z = np.random.standard_normal()
        prices[i] = prices[i - 1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        )

    returns = np.diff(np.log(prices))
    vol = np.std(returns) * np.sqrt(252)

    return {
        "symbol": "SYNTH_STOCK",
        "prices": prices,
        "returns": returns,
        "volatility": vol,
        "current_price": prices[-1],
    }


def _generate_synthetic_crypto_data(
    symbol: str = "BTCUSDT",
    s0: float = 50000.0,
    mu: float = 0.0,
    sigma: float = 0.6,
    n_days: int = 365,
) -> Dict[str, np.ndarray]:
    """Generate synthetic crypto price data with higher volatility."""
    dt = 1.0 / 365
    prices = np.zeros(n_days)
    prices[0] = s0

    np.random.seed(123)
    for i in range(1, n_days):
        z = np.random.standard_normal()
        prices[i] = prices[i - 1] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        )

    returns = np.diff(np.log(prices))
    vol = np.std(returns) * np.sqrt(365)

    return {
        "symbol": symbol,
        "prices": prices,
        "returns": returns,
        "volatility": vol,
        "current_price": prices[-1],
    }


def estimate_implied_volatility(
    prices: np.ndarray,
    window: int = 30,
    annualization: int = 252,
) -> np.ndarray:
    """
    Estimate rolling historical volatility as a proxy for implied vol.

    Parameters
    ----------
    prices : np.ndarray
        Price series.
    window : int
        Rolling window size.
    annualization : int
        Annualization factor (252 for stocks, 365 for crypto).

    Returns
    -------
    vol : np.ndarray
        Rolling annualized volatility.
    """
    log_returns = np.diff(np.log(prices))
    n = len(log_returns)
    vol = np.full(n, np.nan)

    for i in range(window - 1, n):
        vol[i] = np.std(log_returns[i - window + 1 : i + 1]) * np.sqrt(
            annualization
        )

    return vol


def prepare_training_data(
    strike: float = 100.0,
    volatility: float = 0.2,
    maturity: float = 1.0,
    n_interior: int = 10000,
    n_boundary: int = 500,
    n_terminal: int = 500,
    device: str = "cpu",
) -> CollocationData:
    """
    Convenience function to prepare training data with market-calibrated parameters.

    Parameters
    ----------
    strike : float
        Strike price (determines s_max).
    volatility : float
        Volatility for domain sizing.
    maturity : float
        Time to maturity.
    n_interior, n_boundary, n_terminal : int
        Number of collocation points.
    device : str
        Torch device.

    Returns
    -------
    data : CollocationData
    """
    # Set domain boundaries based on strike and volatility
    s_max = strike * np.exp(3.0 * volatility * np.sqrt(maturity))
    s_max = max(s_max, 3.0 * strike)

    return generate_collocation_points(
        s_min=0.0,
        s_max=s_max,
        t_max=maturity,
        n_interior=n_interior,
        n_boundary=n_boundary,
        n_terminal=n_terminal,
        device=device,
        use_sobol=True,
    )
