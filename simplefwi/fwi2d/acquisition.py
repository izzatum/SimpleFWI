"""Bilinear pressure sampling and its exact Euclidean transpose scatter."""
from __future__ import annotations

import math
import torch

from .pml import GridLayout
from .types import BoundaryConfig, Grid2D


class SamplingOperator:
    """Sample physical grids, or compact DOFs if ``boundary`` is supplied.

    Positions are (z,x) in metres. Interpolation coordinates are never
    clipped. Scatter is P^H without an area weight; source_rhs explicitly
    divides by dz*dx. Partial interpolation onto a prescribed zero pressure
    node contributes zero; a point exactly on a Dirichlet boundary is rejected.
    """
    def __init__(self, grid: Grid2D, positions_m, *, boundary: BoundaryConfig | None = None):
        self.grid,self.boundary = grid,boundary
        positions = torch.as_tensor(positions_m,dtype=torch.float64)
        if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] == 0:
            raise ValueError('positions must have nonempty shape [point,2] in (z,x) metres')
        if not bool(torch.all(torch.isfinite(positions))):
            raise ValueError('positions must be finite')
        self.positions_m = tuple(tuple(float(v) for v in row) for row in positions)
        self.n_points = len(self.positions_m)
        layout = GridLayout(grid,boundary) if boundary is not None else None
        self.field_shape = (layout.n_dof,) if layout else tuple(grid.shape)
        indices,weights = [],[]
        nz,nx = grid.shape
        for z,x in self.positions_m:
            coordinates = (z,x)
            fractional = []
            endpoints = []
            for value,origin,spacing,size in zip(coordinates,grid.origin_m,grid.spacing_m,grid.shape):
                upper = origin+spacing*(size-1)
                if value < origin or value > upper:
                    raise ValueError('position lies outside the physical domain; clipping is forbidden')
                at_lower,at_upper = value == origin,value == upper
                endpoints.append((at_lower,at_upper))
                # Preserve exact represented endpoints before subtraction and
                # division can move their fractional index off the boundary.
                # Do not use an isclose tolerance: adjacent interior/outside
                # floats must retain their distinct geometric classification.
                fractional.append(0. if at_lower else float(size-1) if at_upper else (value-origin)/spacing)
            fz,fx = fractional
            if layout:
                (z_lower,z_upper),(x_lower,x_upper) = endpoints
                if ((layout.top_pad == 0 and z_lower) or
                    (boundary.pml_cells == 0 and (z_upper or x_lower or x_upper))):
                    raise ValueError('source/receiver lies on a Dirichlet pressure-release boundary')
            iz,ix = min(math.floor(fz),nz-2),min(math.floor(fx),nx-2)
            wz,wx = fz-iz,fx-ix
            point_indices,point_weights = [],[]
            for zz,xx,w in ((iz,ix,(1-wz)*(1-wx)),(iz,ix+1,(1-wz)*wx),
                            (iz+1,ix,wz*(1-wx)),(iz+1,ix+1,wz*wx)):
                if layout:
                    zz,xx = zz+layout.top_pad,xx+boundary.pml_cells
                    if zz in (0,layout.padded_shape[0]-1) or xx in (0,layout.padded_shape[1]-1):
                        index,w = 0,0.
                    else:
                        index = (zz-1)*(layout.padded_shape[1]-2)+xx-1
                else:
                    index = zz*nx+xx
                point_indices.append(index); point_weights.append(w)
            indices.append(point_indices); weights.append(point_weights)
        self._indices = torch.tensor(indices,dtype=torch.int64)
        self._weights = torch.tensor(weights,dtype=torch.float64)

    def sample(self,u: torch.Tensor) -> torch.Tensor:
        if not (u.is_floating_point() or u.is_complex()):
            raise TypeError('wavefield must have floating or complex dtype')
        nd = len(self.field_shape)
        if tuple(u.shape[-nd:]) != self.field_shape:
            raise ValueError(f'wavefield must end in shape {self.field_shape}')
        flat = u.reshape(*u.shape[:-nd],math.prod(self.field_shape))
        indices = self._indices.to(u.device)
        weights = self._weights.to(device=u.device,dtype=u.real.dtype)
        return (flat[...,indices]*weights).sum(dim=-1)

    def scatter(self,values: torch.Tensor) -> torch.Tensor:
        if not (values.is_floating_point() or values.is_complex()):
            raise TypeError('sample values must have floating or complex dtype')
        if values.ndim < 1 or values.shape[-1] != self.n_points:
            raise ValueError('sample values must end in n_points')
        weights = self._weights.to(device=values.device,dtype=values.real.dtype)
        indices = self._indices.to(values.device).flatten()
        contributions = (values[..., :,None]*weights).flatten(start_dim=-2)
        out = values.new_zeros(*values.shape[:-1],math.prod(self.field_shape))
        out = out.scatter_add(-1,indices.expand_as(contributions),contributions)
        return out.reshape(*values.shape[:-1],*self.field_shape)

    def source_rhs(self,amplitudes: torch.Tensor) -> torch.Tensor:
        """Discrete delta sources: P^H amplitudes/(dz*dx)."""
        return self.scatter(amplitudes)/(self.grid.spacing_m[0]*self.grid.spacing_m[1])
