"""Frozen cubic stretching and differentiable nearest-boundary extension.

Coordinates label nodes. PML adds ``pml_cells`` intervals outside the
physical boundary; the outermost padded nodes have zero pressure.
With exp(-i omega t), ordinary PML has s = 1 + i sigma / omega.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from .types import BoundaryConfig, Grid2D


@dataclass(frozen=True)
class GridLayout:
    grid: Grid2D
    boundary: BoundaryConfig

    @property
    def top_pad(self):
        return 0 if self.boundary.top == 'free_surface' else self.boundary.pml_cells

    @property
    def padded_shape(self):
        nz, nx = self.grid.shape
        p = self.boundary.pml_cells
        return nz + self.top_pad + p, nx + 2*p

    @property
    def n_dof(self):
        nz, nx = self.padded_shape
        return (nz-2)*(nx-2)

    @property
    def physical_slices(self):
        nz, nx = self.grid.shape
        p = self.boundary.pml_cells
        return slice(self.top_pad, self.top_pad+nz), slice(p, p+nx)


def extend_model(m: torch.Tensor, grid: Grid2D, boundary: BoundaryConfig) -> torch.Tensor:
    """Return E(m) on padded nodes; autograd sums repeated boundary entries.

    Nearest extension commutes with v -> 1/v**2, so this is exactly the
    specified nearest-velocity extension in physical slowness coordinates.
    Negative inputs are permitted for model perturbations.
    """
    if not isinstance(m, torch.Tensor) or tuple(m.shape) != tuple(grid.shape):
        raise ValueError('model must be a tensor with physical grid shape [z,x]')
    if m.dtype not in (torch.float32, torch.float64):
        raise TypeError('model must have real float32 or float64 dtype')
    layout = GridLayout(grid, boundary)
    nz, nx = grid.shape
    zz = (torch.arange(layout.padded_shape[0], device=m.device)-layout.top_pad).clamp(0,nz-1)
    xx = (torch.arange(layout.padded_shape[1], device=m.device)-boundary.pml_cells).clamp(0,nx-1)
    return m[zz[:,None], xx[None,:]]


def validate_omega(omega: float) -> float:
    omega = float(omega)
    if not math.isfinite(omega) or omega <= 0:
        raise ValueError('angular frequency omega must be finite and strictly positive')
    return omega


def pml_coefficients(grid: Grid2D, boundary: BoundaryConfig, omega: float,
                     *, dtype=torch.float64, device=None):
    """Return x-face, z-face and centre-mass coefficients, before h factors.

    ax[z,x+1/2] = sz[z]/sx[x+1/2], az[z+1/2,x] = sx[x]/sz[z+1/2],
    b[z,x] = sx[x]*sz[z]. Each stretch is evaluated at its own coordinate,
    rather than averaging the nonlinear ratio of nodal coefficients.
    """
    omega = validate_omega(omega)
    if boundary.alpha != 0:
        raise ValueError('only validated ordinary PML (alpha=0) is implemented')
    layout = GridLayout(grid, boundary)
    nz, nx = layout.padded_shape
    p = boundary.pml_cells

    def stretch(coords, low, high, left_padding):
        if p == 0:
            sigma = torch.zeros_like(coords)
        else:
            left = (low-coords).clamp_min(0) if left_padding else torch.zeros_like(coords)
            distance = torch.maximum(left, (coords-high).clamp_min(0))
            sigma = boundary.sigma_max*(distance/p).pow(3)
        return 1+1j*sigma/omega

    x = torch.arange(nx, dtype=dtype, device=device)
    z = torch.arange(nz, dtype=dtype, device=device)
    sx = stretch(x,p,p+grid.shape[1]-1,True)
    sz = stretch(z,layout.top_pad,layout.top_pad+grid.shape[0]-1,layout.top_pad > 0)
    sxf = stretch(x[:-1]+.5,p,p+grid.shape[1]-1,True)
    szf = stretch(z[:-1]+.5,layout.top_pad,layout.top_pad+grid.shape[0]-1,layout.top_pad > 0)
    return sz[:,None]/sxf[None,:], sx[None,:]/szf[:,None], sz[:,None]*sx[None,:]
