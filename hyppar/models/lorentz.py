"""Lorentz hyperbolic geometry helpers (minimal, Hyper3-CLIP-style)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def lorentz_inner(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Minkowski inner product: -u0*v0 + <u1:, v1:>."""
    spatial = (u[..., 1:] * v[..., 1:]).sum(dim=-1)
    return -u[..., 0] * v[..., 0] + spatial


def project_to_hyperboloid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Force time component so Lorentz norm is -1/c with c absorbed in coords.

    We store points as (x0, x_spatial) on the hyperboloid with <x,x>_L = -1
    when curvature magnitude is folded into the embedding (c=1 units), then
    scale distances by 1/sqrt(c) externally.
    """
    spatial = x[..., 1:]
    x0 = torch.sqrt(torch.clamp(1.0 + (spatial * spatial).sum(dim=-1), min=eps))
    return torch.cat([x0.unsqueeze(-1), spatial], dim=-1)


def exp_map0(v: torch.Tensor, c: float | torch.Tensor = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Exponential map at origin from tangent space R^d to Lorentz H^{d}.

    v: (..., d) Euclidean tangent vector at origin.
    Returns: (..., d+1) Lorentz point on <x,x>_L = -1/c.
    """
    if not torch.is_tensor(c):
        c = torch.tensor(c, device=v.device, dtype=v.dtype)
    c = c.to(device=v.device, dtype=v.dtype)
    sqrt_c = torch.sqrt(c)
    v = v * sqrt_c
    v_norm = torch.clamp(v.norm(dim=-1, keepdim=True), min=eps)
    direction = v / v_norm
    sinh = torch.sinh(v_norm)
    cosh = torch.cosh(v_norm)
    spatial = sinh * direction / sqrt_c
    time = cosh / sqrt_c
    return torch.cat([time, spatial], dim=-1)


def lorentz_distance(
    u: torch.Tensor, v: torch.Tensor, c: float | torch.Tensor = 1.0, eps: float = 1e-5
) -> torch.Tensor:
    """d_H(u,v) = arcosh( -c <u,v>_L ) / sqrt(c)."""
    if not torch.is_tensor(c):
        c = torch.tensor(c, device=u.device, dtype=u.dtype)
    c = c.to(device=u.device, dtype=u.dtype)
    prod = -c * lorentz_inner(u, v)
    prod = torch.clamp(prod, min=1.0 + eps)
    return torch.acosh(prod) / torch.sqrt(c)


class LorentzProjection(nn.Module):
    """Map Euclidean features -> Lorentz hyperboloid via linear + exp map."""

    def __init__(self, in_dim: int, out_dim: int = 128, c: float = 1.0, learnable_c: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        if learnable_c:
            self.log_c = nn.Parameter(torch.tensor(float(c)).log())
        else:
            self.register_buffer("log_c", torch.tensor(float(c)).log())

    @property
    def c(self) -> torch.Tensor:
        return self.log_c.exp().clamp(min=1e-4, max=10.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.linear(x)
        return exp_map0(v, c=self.c)


class EuclideanProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(x), dim=-1)


def euclidean_distance(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return (u - v).pow(2).sum(dim=-1).sqrt()


def lorentz_origin(
    batch_shape: tuple[int, ...] | torch.Size,
    dim: int,
    c: float | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Origin of the hyperboloid: (1/sqrt(c), 0, ..., 0) with ambient dim = dim+1."""
    if not torch.is_tensor(c):
        c = torch.tensor(c, device=device, dtype=dtype)
    c = c.to(device=device, dtype=dtype)
    out = torch.zeros(*batch_shape, dim + 1, device=device, dtype=dtype)
    out[..., 0] = 1.0 / torch.sqrt(c)
    return out


def dist_to_origin(x: torch.Tensor, c: float | torch.Tensor = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Hyperbolic distance from x to the origin."""
    origin = lorentz_origin(x.shape[:-1], x.size(-1) - 1, c, x.device, x.dtype)
    return lorentz_distance(x, origin, c=c, eps=eps)


def entailment_energy(
    premise: torch.Tensor,
    hypothesis: torch.Tensor,
    c: float | torch.Tensor = 1.0,
    K: float = 0.1,
    eps: float = 1e-5,
) -> torch.Tensor:
    """MERU-style cone entailment energy (lower = hypothesis inside premise cone).

    premise: more specific point (gallery image), deeper in hyperbolic space.
    hypothesis: more general point (attribute query), nearer the origin.

    Energy = max(0, exterior_angle(premise; hypothesis) - aperture(premise)).
    """
    # Keep hyperbolic trig in fp32 for AMP stability.
    premise = premise.float()
    hypothesis = hypothesis.float()
    if not torch.is_tensor(c):
        c = torch.tensor(c, device=premise.device, dtype=torch.float32)
    c = c.to(device=premise.device, dtype=torch.float32)
    sqrt_c = torch.sqrt(c)

    ox = dist_to_origin(premise, c=c, eps=eps)
    oy = dist_to_origin(hypothesis, c=c, eps=eps)
    xy = lorentz_distance(premise, hypothesis, c=c, eps=eps)

    num = torch.cosh(sqrt_c * ox) * torch.cosh(sqrt_c * xy) - torch.cosh(sqrt_c * oy)
    den = (torch.sinh(sqrt_c * ox) * torch.sinh(sqrt_c * xy)).clamp(min=eps)
    cos_angle = (num / den).clamp(-1.0 + eps, 1.0 - eps)
    angle = torch.acos(cos_angle)

    sin_aper = (K / torch.sinh(sqrt_c * ox + eps)).clamp(max=1.0 - 1e-4)
    aper = torch.asin(sin_aper)
    return F.relu(angle - aper)


def entailment_distance(
    query: torch.Tensor,
    gallery: torch.Tensor,
    c: float | torch.Tensor = 1.0,
    K: float = 0.1,
    beta: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Retrieval distance: Lorentz d(q,g) + beta * entailment_energy(g entails q)."""
    query = query.float()
    gallery = gallery.float()
    d = lorentz_distance(query, gallery, c=c, eps=eps)
    e = entailment_energy(gallery, query, c=c, K=K, eps=eps)
    return d + beta * e
