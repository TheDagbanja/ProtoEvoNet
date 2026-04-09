"""
Fisher–Rao geodesic distance for Gamma distributions.

The Fisher information matrix of Gamma(k, θ) with respect to the natural
parameters (k, θ) is:

    G(k, θ) = | ψ₁(k)    -1/θ |
              | -1/θ    k/θ²  |

where ψ₁ = trigamma is the first polygamma function.

The Fisher–Rao (information-geometric) distance between two Gamma
distributions p₁ = Gamma(k₁, θ₁) and p₂ = Gamma(k₂, θ₂) is the length of
the geodesic curve on the statistical manifold.  A closed-form expression
does not exist in general; this module implements two approximations:

1. **Infinitesimal (first-order) approximation**:

        d²_FR ≈ Δpᵀ G(k̄, θ̄) Δp,   Δp = (k₁-k₂, θ₁-θ₂), (k̄, θ̄) midpoint

   This is exact when the two distributions are close together.

2. **Numerical geodesic** (more accurate, slower): integrate the metric
   along the straight line in parameter space (not the true geodesic, but a
   good approximation for moderate distances).

Both implementations are fully differentiable.
"""

from __future__ import annotations

import torch
from torch.special import polygamma


# ---------------------------------------------------------------------------
# Fisher information matrix utilities
# ---------------------------------------------------------------------------

def fisher_info_matrix(k: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Compute the 2×2 Fisher information matrix for Gamma(k, θ).

    Parameters
    ----------
    k:
        Shape parameter tensor, any shape ``(*batch,)``.
    theta:
        Scale parameter tensor, same shape as *k*.

    Returns
    -------
    torch.Tensor
        Shape ``(*batch, 2, 2)`` Fisher information matrices.
    """
    k = k.clamp(min=1e-8)
    theta = theta.clamp(min=1e-8)

    psi1_k = polygamma(1, k)              # trigamma: (*batch,)
    off_diag = -1.0 / theta              # (*batch,)
    diag_kk = psi1_k                     # (*batch,)
    diag_tt = k / (theta ** 2)           # (*batch,)

    # Stack into (*batch, 2, 2)
    batch_shape = k.shape
    G = torch.zeros(*batch_shape, 2, 2, device=k.device, dtype=k.dtype)
    G[..., 0, 0] = diag_kk
    G[..., 0, 1] = off_diag
    G[..., 1, 0] = off_diag
    G[..., 1, 1] = diag_tt
    return G


# ---------------------------------------------------------------------------
# Infinitesimal Fisher–Rao distance
# ---------------------------------------------------------------------------

def fisher_rao_distance_approx(
    k1: torch.Tensor,
    theta1: torch.Tensor,
    k2: torch.Tensor,
    theta2: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    First-order (infinitesimal) Fisher–Rao distance between Gamma pairs.

    Uses the Fisher metric evaluated at the parameter midpoint.

    Parameters
    ----------
    k1, theta1:
        Parameters of the first distribution, shape ``(*batch,)``.
    k2, theta2:
        Parameters of the second distribution, shape ``(*batch,)``.
    eps:
        Numerical stability floor.

    Returns
    -------
    torch.Tensor
        Non-negative distance values, shape ``(*batch,)``.
    """
    # Midpoint
    k_mid = 0.5 * (k1 + k2)
    t_mid = 0.5 * (theta1 + theta2)

    G = fisher_info_matrix(k_mid, t_mid)  # (*batch, 2, 2)

    # Difference vector (*batch, 2, 1)
    dk = (k1 - k2).unsqueeze(-1)
    dt = (theta1 - theta2).unsqueeze(-1)
    delta = torch.cat([dk, dt], dim=-1).unsqueeze(-1)  # (*batch, 2, 1)

    # d² = deltaᵀ G delta
    Gd = torch.matmul(G, delta)        # (*batch, 2, 1)
    d_sq = torch.matmul(delta.transpose(-2, -1), Gd).squeeze(-1).squeeze(-1)
    d_sq = d_sq.clamp(min=0.0)

    return torch.sqrt(d_sq + eps)


# ---------------------------------------------------------------------------
# Numerical geodesic approximation (line-integral along straight line)
# ---------------------------------------------------------------------------

def fisher_rao_distance_numerical(
    k1: torch.Tensor,
    theta1: torch.Tensor,
    k2: torch.Tensor,
    theta2: torch.Tensor,
    n_steps: int = 10,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Numerically approximate the Fisher–Rao geodesic distance by integrating
    the Riemannian metric along the straight-line path in parameter space.

        d_FR ≈ ∫₀¹ sqrt(ṗ(t)ᵀ G(p(t)) ṗ(t)) dt,   p(t) = (1-t)p₁ + t·p₂

    Parameters
    ----------
    k1, theta1, k2, theta2:
        Gamma parameters, shape ``(*batch,)``.
    n_steps:
        Number of quadrature points.
    eps:
        Numerical stability floor.

    Returns
    -------
    torch.Tensor
        Approximate geodesic distance, shape ``(*batch,)``.
    """
    # Velocity (constant along the straight line)
    dk = k2 - k1       # (*batch,)
    dt = theta2 - theta1   # (*batch,)

    ts = torch.linspace(0.0, 1.0, n_steps, device=k1.device, dtype=k1.dtype)
    step_size = 1.0 / (n_steps - 1) if n_steps > 1 else 1.0

    total = torch.zeros_like(k1)

    for i, t in enumerate(ts):
        k_t = (k1 + t * dk).clamp(min=eps)
        theta_t = (theta1 + t * dt).clamp(min=eps)

        G = fisher_info_matrix(k_t, theta_t)  # (*batch, 2, 2)

        # velocity vector (*batch, 2, 1)
        vel = torch.stack([dk, dt], dim=-1).unsqueeze(-1)

        # quadratic form
        Gv = torch.matmul(G, vel)           # (*batch, 2, 1)
        metric_val = torch.matmul(vel.transpose(-2, -1), Gv)
        metric_val = metric_val.squeeze(-1).squeeze(-1).clamp(min=0.0)

        ds = torch.sqrt(metric_val + eps)

        # Trapezoidal weights
        if i == 0 or i == n_steps - 1:
            total = total + 0.5 * ds * step_size
        else:
            total = total + ds * step_size

    return total


# ---------------------------------------------------------------------------
# Convenience wrapper (nn.Module-free, used by PhysicsLayer)
# ---------------------------------------------------------------------------

class FisherRaoDistance(torch.nn.Module):
    """
    Differentiable Fisher–Rao distance module for Gamma distributions.

    Parameters
    ----------
    method:
        ``"approx"`` for the fast infinitesimal approximation or
        ``"numerical"`` for the line-integral version.
    n_steps:
        Quadrature steps (only used when ``method="numerical"``).
    eps:
        Numerical stability floor.
    use_fisher_rao:
        If ``False``, returns Euclidean distance in (k, θ) space (ablation).
    """

    def __init__(
        self,
        method: str = "approx",
        n_steps: int = 10,
        eps: float = 1e-8,
        use_fisher_rao: bool = True,
    ) -> None:
        super().__init__()
        self.method = method
        self.n_steps = n_steps
        self.eps = eps
        self.use_fisher_rao = use_fisher_rao

    def forward(
        self,
        k1: torch.Tensor,
        theta1: torch.Tensor,
        k2: torch.Tensor,
        theta2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distance between pairs of Gamma distributions.

        Parameters
        ----------
        k1, theta1, k2, theta2:
            Gamma parameters, shape ``(*batch,)``.

        Returns
        -------
        torch.Tensor
            Distances, shape ``(*batch,)``.
        """
        if not self.use_fisher_rao:
            # Ablation: plain Euclidean distance in parameter space
            return torch.sqrt(
                (k1 - k2) ** 2 + (theta1 - theta2) ** 2 + self.eps
            )

        if self.method == "numerical":
            return fisher_rao_distance_numerical(
                k1, theta1, k2, theta2, n_steps=self.n_steps, eps=self.eps
            )
        return fisher_rao_distance_approx(
            k1, theta1, k2, theta2, eps=self.eps
        )

    def extra_repr(self) -> str:
        return (
            f"method={self.method}, use_fisher_rao={self.use_fisher_rao}"
        )
