"""
PINN Model for American Option Pricing
=======================================

Implements a Physics-Informed Neural Network that solves the American option
pricing problem using the penalty method for the free boundary.

The American option PDE with free boundary is:
    max(dV/dt + 0.5*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V, payoff(S) - V) = 0

This is reformulated using a penalty approach:
    dV/dt + 0.5*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V + lambda*max(payoff(S) - V, 0) = 0

where lambda is a large penalty parameter that enforces the early exercise constraint.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict


class AmericanOptionPINN(nn.Module):
    """
    Physics-Informed Neural Network for American option pricing.

    The network takes (S, t) as input and outputs V(S, t) — the option value.
    The loss function includes:
      1. PDE residual (Black-Scholes operator)
      2. Boundary conditions (at S=0, S->inf, t=T)
      3. Early exercise penalty (free boundary condition)

    Parameters
    ----------
    hidden_layers : list of int
        Number of neurons in each hidden layer.
    activation : str
        Activation function: 'tanh', 'relu', 'softplus', 'gelu'.
    option_type : str
        'put' or 'call' for American put/call.
    """

    def __init__(
        self,
        hidden_layers: list = None,
        activation: str = "tanh",
        option_type: str = "put",
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [64, 64, 64, 64]

        self.option_type = option_type

        # Build network layers
        layers = []
        input_dim = 2  # (S, t)
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "softplus":
                layers.append(nn.Softplus())
            elif activation == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            input_dim = hidden_dim

        # Output layer: option value V(S, t) >= 0
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Softplus())  # ensure non-negative output

        self.network = nn.Sequential(*layers)

        # Initialize weights using Xavier initialization
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict option value V(S, t).

        Parameters
        ----------
        S : torch.Tensor
            Spot price (normalized), shape (batch,)
        t : torch.Tensor
            Time to maturity, shape (batch,)

        Returns
        -------
        V : torch.Tensor
            Option value, shape (batch, 1)
        """
        x = torch.stack([S, t], dim=-1)
        return self.network(x)


class AmericanPINNPricer:
    """
    Full American option pricer using PINN with penalty method.

    Handles:
      - PDE residual loss (Black-Scholes operator)
      - Boundary condition losses
      - Early exercise penalty
      - Free boundary detection
      - Greeks computation via autograd

    Parameters
    ----------
    strike : float
        Strike price K.
    risk_free_rate : float
        Risk-free interest rate r.
    volatility : float
        Volatility sigma.
    maturity : float
        Time to maturity T (in years).
    option_type : str
        'put' or 'call'.
    s_max : float
        Maximum spot price for the domain.
    penalty_lambda : float
        Penalty parameter for early exercise constraint.
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(
        self,
        strike: float = 100.0,
        risk_free_rate: float = 0.05,
        volatility: float = 0.2,
        maturity: float = 1.0,
        option_type: str = "put",
        s_max: float = 300.0,
        penalty_lambda: float = 1000.0,
        hidden_layers: list = None,
        device: str = "cpu",
    ):
        self.K = strike
        self.r = risk_free_rate
        self.sigma = volatility
        self.T = maturity
        self.option_type = option_type
        self.s_max = s_max
        self.penalty_lambda = penalty_lambda
        self.device = torch.device(device)

        if hidden_layers is None:
            hidden_layers = [64, 64, 64, 64]

        self.model = AmericanOptionPINN(
            hidden_layers=hidden_layers,
            activation="tanh",
            option_type=option_type,
        ).to(self.device)

    def payoff(self, S: torch.Tensor) -> torch.Tensor:
        """Compute option payoff."""
        if self.option_type == "put":
            return torch.clamp(self.K - S, min=0.0)
        else:
            return torch.clamp(S - self.K, min=0.0)

    def pde_residual(
        self, S: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Black-Scholes PDE residual.

        The PDE: dV/dt + 0.5*sigma^2*S^2*d2V/dS2 + r*S*dV/dS - r*V = 0

        Uses automatic differentiation to compute derivatives.

        Parameters
        ----------
        S : torch.Tensor
            Spot prices requiring grad, shape (N,)
        t : torch.Tensor
            Times requiring grad, shape (N,)

        Returns
        -------
        residual : torch.Tensor
            PDE residual at each point, shape (N, 1)
        """
        S.requires_grad_(True)
        t.requires_grad_(True)

        V = self.model(S, t)

        # First derivatives via autograd
        dV = torch.autograd.grad(
            V, [S, t],
            grad_outputs=torch.ones_like(V),
            create_graph=True,
            retain_graph=True,
        )
        dV_dS = dV[0].unsqueeze(-1)  # (N, 1)
        dV_dt = dV[1].unsqueeze(-1)  # (N, 1)

        # Second derivative d2V/dS2
        d2V_dS2 = torch.autograd.grad(
            dV_dS, S,
            grad_outputs=torch.ones_like(dV_dS),
            create_graph=True,
            retain_graph=True,
        )[0].unsqueeze(-1)  # (N, 1)

        S_col = S.unsqueeze(-1)  # (N, 1)

        # Black-Scholes PDE: dV/dt + 0.5*sig^2*S^2*V_SS + r*S*V_S - r*V
        residual = (
            dV_dt
            + 0.5 * self.sigma ** 2 * S_col ** 2 * d2V_dS2
            + self.r * S_col * dV_dS
            - self.r * V
        )

        return residual

    def compute_loss(
        self,
        S_interior: torch.Tensor,
        t_interior: torch.Tensor,
        S_boundary: torch.Tensor,
        t_boundary: torch.Tensor,
        S_terminal: torch.Tensor,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total PINN loss for American options.

        Components:
          1. PDE residual loss on interior points
          2. Terminal condition loss at t = T
          3. Boundary condition losses at S = 0 and S = S_max
          4. Early exercise penalty loss

        Parameters
        ----------
        S_interior : torch.Tensor
            Interior spot prices, shape (N_int,)
        t_interior : torch.Tensor
            Interior times, shape (N_int,)
        S_boundary : torch.Tensor
            Boundary spot prices, shape (N_bnd,)
        t_boundary : torch.Tensor
            Boundary times, shape (N_bnd,)
        S_terminal : torch.Tensor
            Terminal spot prices, shape (N_term,)
        weights : dict, optional
            Loss component weights.

        Returns
        -------
        total_loss : torch.Tensor
            Weighted sum of all loss components.
        loss_dict : dict
            Individual loss component values.
        """
        if weights is None:
            weights = {
                "pde": 1.0,
                "terminal": 10.0,
                "boundary": 5.0,
                "penalty": 1.0,
            }

        # 1. PDE residual loss on interior
        residual = self.pde_residual(S_interior, t_interior)
        loss_pde = torch.mean(residual ** 2)

        # 2. Terminal condition: V(S, T) = payoff(S)
        t_T = torch.full_like(S_terminal, self.T)
        V_terminal = self.model(S_terminal, t_T)
        payoff_terminal = self.payoff(S_terminal).unsqueeze(-1)
        loss_terminal = torch.mean((V_terminal - payoff_terminal) ** 2)

        # 3. Boundary conditions
        if self.option_type == "put":
            # V(0, t) = K * exp(-r*(T-t)) for put
            S_zero = torch.zeros_like(t_boundary)
            V_at_zero = self.model(S_zero, t_boundary)
            target_zero = self.K * torch.exp(
                -self.r * (self.T - t_boundary)
            ).unsqueeze(-1)
            loss_bc_low = torch.mean((V_at_zero - target_zero) ** 2)

            # V(S_max, t) -> 0 for put as S -> inf
            S_high = torch.full_like(t_boundary, self.s_max)
            V_at_high = self.model(S_high, t_boundary)
            loss_bc_high = torch.mean(V_at_high ** 2)
        else:
            # V(0, t) = 0 for call
            S_zero = torch.zeros_like(t_boundary)
            V_at_zero = self.model(S_zero, t_boundary)
            loss_bc_low = torch.mean(V_at_zero ** 2)

            # V(S_max, t) ~ S_max - K*exp(-r*(T-t)) for call
            S_high = torch.full_like(t_boundary, self.s_max)
            V_at_high = self.model(S_high, t_boundary)
            target_high = (
                self.s_max
                - self.K * torch.exp(-self.r * (self.T - t_boundary))
            ).unsqueeze(-1)
            loss_bc_high = torch.mean((V_at_high - target_high) ** 2)

        loss_boundary = loss_bc_low + loss_bc_high

        # 4. Early exercise penalty: penalize V < payoff
        V_interior = self.model(S_interior.detach(), t_interior.detach())
        payoff_interior = self.payoff(S_interior.detach()).unsqueeze(-1)
        violation = torch.clamp(payoff_interior - V_interior, min=0.0)
        loss_penalty = self.penalty_lambda * torch.mean(violation ** 2)

        # Total weighted loss
        total_loss = (
            weights["pde"] * loss_pde
            + weights["terminal"] * loss_terminal
            + weights["boundary"] * loss_boundary
            + weights["penalty"] * loss_penalty
        )

        loss_dict = {
            "pde": loss_pde.item(),
            "terminal": loss_terminal.item(),
            "boundary": loss_boundary.item(),
            "penalty": loss_penalty.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_dict

    def price(
        self, S: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """
        Price American options at given (S, t) points.

        Parameters
        ----------
        S : np.ndarray
            Spot prices.
        t : np.ndarray
            Times to maturity.

        Returns
        -------
        prices : np.ndarray
            Option prices.
        """
        self.model.eval()
        with torch.no_grad():
            S_t = torch.tensor(S, dtype=torch.float32, device=self.device)
            t_t = torch.tensor(t, dtype=torch.float32, device=self.device)
            V = self.model(S_t, t_t).cpu().numpy().flatten()
        return V

    def find_exercise_boundary(
        self,
        t_values: np.ndarray,
        s_range: Tuple[float, float] = (0.5, 200.0),
        n_points: int = 1000,
        tol: float = 0.01,
    ) -> np.ndarray:
        """
        Find the optimal early exercise boundary S*(t).

        For a put option, the exercise boundary is the highest S where
        V(S, t) = K - S (payoff equals option value).

        Parameters
        ----------
        t_values : np.ndarray
            Time values at which to find the boundary.
        s_range : tuple
            (S_min, S_max) range to search.
        n_points : int
            Number of points in S grid for search.
        tol : float
            Tolerance for boundary detection.

        Returns
        -------
        boundary : np.ndarray
            Exercise boundary S*(t) for each t value.
        """
        self.model.eval()
        S_grid = np.linspace(s_range[0], s_range[1], n_points)
        boundary = np.zeros(len(t_values))

        with torch.no_grad():
            for i, t_val in enumerate(t_values):
                S_t = torch.tensor(
                    S_grid, dtype=torch.float32, device=self.device
                )
                t_t = torch.full_like(S_t, t_val)
                V = self.model(S_t, t_t).cpu().numpy().flatten()

                if self.option_type == "put":
                    intrinsic = np.maximum(self.K - S_grid, 0.0)
                else:
                    intrinsic = np.maximum(S_grid - self.K, 0.0)

                # Find where option value approximately equals intrinsic value
                diff = np.abs(V - intrinsic)
                # For put: find the highest S where they are close
                # (in the exercise region)
                exercise_mask = (intrinsic > tol) & (diff < tol * self.K)

                if np.any(exercise_mask):
                    if self.option_type == "put":
                        boundary[i] = S_grid[exercise_mask].max()
                    else:
                        boundary[i] = S_grid[exercise_mask].min()
                else:
                    # Approximate: find minimum diff where intrinsic > 0
                    mask = intrinsic > tol
                    if np.any(mask):
                        idx = np.argmin(diff[mask])
                        boundary[i] = S_grid[mask][idx]
                    else:
                        boundary[i] = self.K

        return boundary

    def save(self, path: str):
        """Save model state dict."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "params": {
                    "strike": self.K,
                    "risk_free_rate": self.r,
                    "volatility": self.sigma,
                    "maturity": self.T,
                    "option_type": self.option_type,
                    "s_max": self.s_max,
                    "penalty_lambda": self.penalty_lambda,
                },
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model state dict."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from {path}")


def create_pricer(
    strike: float = 100.0,
    risk_free_rate: float = 0.05,
    volatility: float = 0.2,
    maturity: float = 1.0,
    option_type: str = "put",
    device: str = "cpu",
) -> AmericanPINNPricer:
    """Factory function to create a PINN pricer with sensible defaults."""
    return AmericanPINNPricer(
        strike=strike,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        maturity=maturity,
        option_type=option_type,
        s_max=3.0 * strike,
        penalty_lambda=1000.0,
        hidden_layers=[64, 64, 64, 64],
        device=device,
    )
