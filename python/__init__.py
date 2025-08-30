"""
PINN American Options
=====================

Physics-Informed Neural Network for pricing American options
with early exercise boundary detection.

Modules:
    american_pinn  - PINN model with penalty method for free boundary
    train          - Training loop with free boundary handling
    data_loader    - Synthetic + real market data (stocks + Bybit crypto)
    lsm_benchmark  - Longstaff-Schwartz Monte Carlo benchmark
    greeks         - Greeks via automatic differentiation
    visualize      - Exercise boundary + option surface visualization
    backtest       - Trading strategy backtest
"""

__version__ = "0.1.0"
__author__ = "ML Trading Examples"
