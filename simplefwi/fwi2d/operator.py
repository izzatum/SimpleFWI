"""Matched conservative tensor and sparse Helmholtz operators.

The compact vector is the row-major flattening of padded[1:-1,1:-1].
Every outer padded node is homogeneous Dirichlet; with a free surface the
top physical row is precisely the top padded row and is eliminated too.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from .pml import GridLayout, extend_model, pml_coefficients, validate_omega
from .types import BoundaryConfig, Grid2D


class HelmholtzOperator:
    def __init__(self, grid: Grid2D, boundary: BoundaryConfig):
        self.grid, self.boundary = grid, boundary
        self.layout = GridLayout(grid,boundary)
        self.padded_shape = self.layout.padded_shape
        self.n_dof = self.layout.n_dof
        if min(self.padded_shape) < 3:
            raise ValueError('computational domain must have interior pressure nodes')
        if boundary.alpha != 0:
            raise ValueError('only validated ordinary PML (alpha=0) is implemented')

    def pack(self, u: torch.Tensor) -> torch.Tensor:
        """Restrict [...,z_padded,x_padded] to [...,n_dof]."""
        if tuple(u.shape[-2:]) != self.padded_shape:
            raise ValueError('padded wavefield shape mismatch')
        return u[...,1:-1,1:-1].reshape(*u.shape[:-2],self.n_dof)

    def unpack(self, u: torch.Tensor) -> torch.Tensor:
        """Inject [...,n_dof] into padded arrays with exact zero boundaries."""
        if u.ndim < 1 or u.shape[-1] != self.n_dof:
            raise ValueError('compact wavefield must have last axis n_dof')
        nz,nx = self.padded_shape
        return F.pad(u.reshape(*u.shape[:-1],nz-2,nx-2),(1,1,1,1))

    def physical(self, u: torch.Tensor) -> torch.Tensor:
        zs,xs = self.layout.physical_slices
        return self.unpack(u)[...,zs,xs]

    def mass_weights(self, omega: float, *, dtype=torch.float64, device=None) -> torch.Tensor:
        """Compact omega**2 sx sz; model-independent, differentiable products."""
        _,_,mass = pml_coefficients(self.grid,self.boundary,omega,dtype=dtype,device=device)
        return self.pack(mass)*float(omega)**2

    def mass_diagonal(self, m: torch.Tensor, omega: float) -> torch.Tensor:
        return self.mass_weights(omega,dtype=m.dtype,device=m.device)*self.pack(extend_model(m,self.grid,self.boundary))

    def model_derivative(self, dm: torch.Tensor, omega: float, u: torch.Tensor) -> torch.Tensor:
        """D_m A[dm]u, including every extended boundary-cell dependency."""
        return self.mass_diagonal(dm,omega)*u

    def _validate_model(self,m):
        if not isinstance(m,torch.Tensor) or tuple(m.shape) != self.grid.shape:
            raise ValueError('model must have physical shape [z,x]')
        if m.dtype not in (torch.float32,torch.float64):
            raise TypeError('physical slowness squared must be real float32 or float64')
        if not bool(torch.all(torch.isfinite(m) & (m>0))):
            raise ValueError('physical slowness squared must be finite and positive')

    def apply(self, m: torch.Tensor, omega: float, u: torch.Tensor, *, adjoint=False) -> torch.Tensor:
        self._validate_model(m)
        if u.device != m.device:
            raise ValueError('model and wavefield must be on the same device')
        wanted = torch.complex128 if m.dtype == torch.float64 else torch.complex64
        if u.dtype != wanted:
            raise TypeError(f'wavefield dtype must be {wanted} for model dtype {m.dtype}')
        full = self.unpack(u)
        ax,az,b = pml_coefficients(self.grid,self.boundary,omega,dtype=m.dtype,device=m.device)
        if adjoint:
            ax,az,b = ax.conj(),az.conj(),b.conj()
        dz,dx = self.grid.spacing_m
        fx = ax*(full[..., :,1:]-full[..., :,:-1])/dx**2
        fz = az*(full[...,1:,:]-full[...,:-1,:])/dz**2
        divergence = fx[...,1:-1,1:]-fx[...,1:-1,:-1] + fz[...,1:,1:-1]-fz[...,:-1,1:-1]
        mass = float(omega)**2*b[1:-1,1:-1]*extend_model(m,self.grid,self.boundary)[1:-1,1:-1]
        result = divergence + mass*full[...,1:-1,1:-1]
        return result.reshape(*u.shape[:-1],self.n_dof)

    def assemble(self, m, omega: float) -> sp.csc_matrix:
        """CPU sparse reference, using the documented face/centre coefficients.

        Assembly deliberately detaches model values; differentiable operators
        use apply/model_derivative. No implicit device transfer is performed.
        """
        if not isinstance(m,torch.Tensor):
            m = torch.as_tensor(m)
        self._validate_model(m)
        if m.device.type != 'cpu':
            raise ValueError('SciPy assembly requires an explicit CPU model')
        omega = validate_omega(omega)
        ax,az,_ = pml_coefficients(self.grid,self.boundary,omega,dtype=m.dtype,device=m.device)
        dz,dx = self.grid.spacing_m
        ax,az = ax.numpy()/dx**2,az.numpy()/dz**2
        nz,nx = self.padded_shape
        rows,cols,values = [],[],[]
        mass = self.mass_diagonal(m,omega).detach().numpy()
        for z in range(1,nz-1):
            for x in range(1,nx-1):
                k = (z-1)*(nx-2)+x-1
                rows.append(k); cols.append(k)
                values.append(mass[k]-ax[z,x-1]-ax[z,x]-az[z-1,x]-az[z,x])
                for zz,xx,value in ((z,x-1,ax[z,x-1]),(z,x+1,ax[z,x]),
                                    (z-1,x,az[z-1,x]),(z+1,x,az[z,x])):
                    if 0 < zz < nz-1 and 0 < xx < nx-1:
                        rows.append(k); cols.append((zz-1)*(nx-2)+xx-1); values.append(value)
        return sp.csc_matrix((values,(rows,cols)),shape=(self.n_dof,self.n_dof))
