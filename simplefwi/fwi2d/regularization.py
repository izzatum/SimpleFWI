"""Physical slowness smoothness with cell-area-weighted spatial differences."""
from __future__ import annotations

import math
import torch

from .types import Grid2D


class Regularizer:
    """beta/2 times squared forward x/z differences of m-prior_m.

    Each interior edge is weighted by dz*dx. No differences cross opposite
    edges or extend outside the physical grid (natural/no-flux boundaries).
    The prior is fixed, with its device and precision chosen by the caller.
    """
    def __init__(self, grid: Grid2D, beta: float, prior_m: torch.Tensor):
        self.grid = grid
        self.beta = float(beta)
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError('beta must be finite and nonnegative')
        self._validate(prior_m)
        self.prior_m = prior_m.detach().clone()

    def _validate(self, m):
        if not isinstance(m, torch.Tensor) or tuple(m.shape) != self.grid.shape:
            raise ValueError('regularizer model/prior must have physical shape [z,x]')
        if m.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(m).all()):
            raise ValueError('regularizer model/prior must be finite real float32 or float64')

    def value(self, m: torch.Tensor) -> torch.Tensor:
        self._validate(m)
        if m.device != self.prior_m.device or m.dtype != self.prior_m.dtype:
            raise ValueError('model and prior must share explicit precision and device')
        delta = m-self.prior_m
        dz, dx = self.grid.spacing_m
        gz = (delta[1:, :]-delta[:-1, :])/dz
        gx = (delta[:, 1:]-delta[:, :-1])/dx
        return (self.beta/2)*(dz*dx)*(gz.square().sum()+gx.square().sum())
