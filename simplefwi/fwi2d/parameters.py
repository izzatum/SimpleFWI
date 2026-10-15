"""Bounded native PyTorch parameter containers for physical velocity [z, x]."""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn


class VelocityParameters(nn.Module):
    """Map an unconstrained 2D parameter to bounded SI velocity.

    Only active cells participate in the mapping. Frozen background values are
    copied at construction and are independent of optimizer weight decay. The
    sparse route produces sparse gradients, without promising sparse PDE work.
    """
    def __init__(self, initial_velocity, bounds, active_mask=None, *, representation='dense'):
        super().__init__()
        if isinstance(initial_velocity, torch.Tensor):
            # Explicit tensor precision/device is caller-owned; validate below.
            velocity = initial_velocity
        else:
            # Reject complex arrays before requesting a real dtype: PyTorch
            # otherwise discards their imaginary part with only a warning.
            if np.iscomplexobj(initial_velocity):
                raise ValueError('initial_velocity must be a real array')
            # Construct directly in float64. Constructing at the default dtype
            # first would irreversibly round Python decimals through float32.
            velocity = torch.as_tensor(initial_velocity, dtype=torch.float64)
        if (velocity.ndim != 2 or velocity.numel() == 0 or
                velocity.dtype not in (torch.float32, torch.float64) or
                not bool(torch.isfinite(velocity).all())):
            raise ValueError('initial_velocity must be a finite real float32/float64 2D array')
        if len(bounds) != 2:
            raise ValueError('bounds must be (minimum_velocity, maximum_velocity)')
        lower, upper = map(float, bounds)
        if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower < upper):
            raise ValueError('velocity bounds must be finite, positive, and increasing')
        if active_mask is None:
            mask = torch.ones_like(velocity, dtype=torch.bool)
        else:
            mask = torch.as_tensor(active_mask, device=velocity.device)
            if mask.dtype != torch.bool or mask.shape != velocity.shape:
                raise ValueError('active_mask must be boolean with the physical grid shape')
        if not bool(mask.any()):
            raise ValueError('active_mask must select at least one cell')
        if not bool(((velocity >= lower) & (velocity <= upper)).all()):
            raise ValueError('initial velocity, including frozen cells, must lie within bounds')
        if not bool(((velocity[mask] > lower) & (velocity[mask] < upper)).all()):
            raise ValueError('active initial velocities must be strictly inside bounds')
        if representation not in ('dense', 'sparse_embedding'):
            raise ValueError('representation must be dense or sparse_embedding')
        self.bounds = (lower, upper)
        self.representation = representation
        self.register_buffer('background', velocity.detach().clone())
        self.register_buffer('active_mask', mask.detach().clone())
        fraction = (velocity.detach()-lower)/(upper-lower)
        fraction = torch.where(mask, fraction, torch.full_like(fraction, 0.5))
        theta = torch.logit(fraction)
        if representation == 'dense':
            self.theta = nn.Parameter(theta)
        else:
            self.embedding = nn.Embedding(int(mask.sum()), 1, sparse=True,
                                           device=velocity.device, dtype=velocity.dtype)
            with torch.no_grad():
                self.embedding.weight.copy_(theta[mask, None])
            self.register_buffer('active_indices', torch.arange(int(mask.sum()), device=velocity.device))

    def velocity(self):
        lower, upper = self.bounds
        if self.representation == 'dense':
            mapped = lower+(upper-lower)*torch.sigmoid(self.theta)
            return torch.where(self.active_mask, mapped, self.background)
        theta = self.embedding(self.active_indices).squeeze(-1)
        mapped = lower+(upper-lower)*torch.sigmoid(theta)
        return self.background.masked_scatter(self.active_mask, mapped)

    def slowness_squared(self):
        return self.velocity().reciprocal().square()

    @property
    def raw_parameter(self):
        """Dense theta [z,x] or the active sparse embedding weight [active,1]."""
        return self.theta if self.representation == 'dense' else self.embedding.weight

    def slowness_squared_from(self, theta):
        """Pure dense transform for exact theta Hessians, including chain terms.

        Sparse models use physical-coordinate curvature instead of claiming
        sparse native higher-order autodiff support. This method never mutates
        the model or optimizer parameters.
        """
        if self.representation != 'dense':
            raise ValueError('slowness_squared_from requires the dense representation')
        if (not isinstance(theta, torch.Tensor) or theta.shape != self.theta.shape or
                theta.dtype != self.theta.dtype or theta.device != self.theta.device):
            raise ValueError('theta must match the dense parameter shape, dtype, and device')
        lower, upper = self.bounds
        v = torch.where(self.active_mask,
                        lower+(upper-lower)*torch.sigmoid(theta), self.background)
        return v.reciprocal().square()

    def diagnostics(self):
        with torch.no_grad():
            v = self.velocity()
            lower, upper = self.bounds
            f = (v[self.active_mask]-lower)/(upper-lower)
            return {'velocity_min_m_s': float(v.min()), 'velocity_max_m_s': float(v.max()),
                    'active_cells': int(self.active_mask.sum()),
                    'saturated_cells': int(((f < 1e-6) | (f > 1-1e-6)).sum()),
                    'representation': self.representation}
