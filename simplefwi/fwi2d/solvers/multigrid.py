"""Bounded geometric V-cycles for the conservative shifted Helmholtz operator.

For exp(-i omega t), B=A+i*shift*diag(omega**2*sx*sz*E(m)). The physical A
is never changed. Geometric bilinear interpolation P uses nested padded nodes
and eliminates all Dirichlet endpoints (including the physical free surface).
Restriction is exactly P^H and every coarse operator is B_c=P^H B_f P, not a
rediscretization with rounded PML thickness. Nested two-to-one intervals and a
possible final one-cell interval retain a nine-point Galerkin stencil. Nine
color probes recover those coefficients from the actual parent action in O(N)
storage. Only the explicitly capped coarsest grid has a dense device LU.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import time

import torch
import torch.nn.functional as F

from ..operator import HelmholtzOperator
from ..types import SolveReport
from .base import SolverFailure


@dataclass(frozen=True)
class MultigridConfig:
    """The shift and hierarchy limits must be selected explicitly by callers."""
    shift: float
    max_levels: int
    coarse_max_dof: int
    pre_smooth: int = 2
    post_smooth: int = 2
    damping: float = 2/3
    memory_budget_bytes: int = 256*1024**2

    def __post_init__(self):
        if not math.isfinite(self.shift) or self.shift <= 0:
            raise ValueError('shift must be finite and strictly positive')
        if not math.isfinite(self.damping) or not 0 < self.damping <= 1:
            raise ValueError('damping must lie in (0,1]')
        for name in ('max_levels', 'coarse_max_dof', 'memory_budget_bytes'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('pre_smooth', 'post_smooth'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')

    def to_dict(self):
        return asdict(self)


def _coarse_nodes(length):
    if length <= 3:
        return list(range(length))
    indices = list(range(0, length, 2))
    if indices[-1] != length-1:
        indices.append(length-1)
    return indices


class _Transfer:
    def __init__(self, fine_shape, *, dtype, device):
        self.fine_shape = fine_shape
        self.coarse_shape = tuple(len(_coarse_nodes(n)) for n in fine_shape)
        self.axes = []
        for length in fine_shape:
            nodes = torch.tensor(_coarse_nodes(length), device=device)
            locations = torch.arange(length, device=device)
            lower = (torch.searchsorted(nodes, locations, right=True)-1).clamp(0, len(nodes)-2)
            weight = ((locations-nodes[lower])/(nodes[lower+1]-nodes[lower])).to(dtype)
            self.axes.append((lower, weight))

    def prolong_padded(self, coarse):
        cz, cx = self.coarse_shape
        full = F.pad(coarse.reshape(*coarse.shape[:-1], cz-2, cx-2), (1,1,1,1))
        ix, wx = self.axes[1]
        along_x = full[..., ix]*(1-wx) + full[..., ix+1]*wx
        iz, wz = self.axes[0]
        return along_x[..., iz, :]*(1-wz[:, None]) + along_x[..., iz+1, :]*wz[:, None]

    def prolong(self, coarse):
        full = self.prolong_padded(coarse)
        return full[..., 1:-1, 1:-1].reshape(*coarse.shape[:-1], -1)

    def restrict(self, fine):
        fz, fx = self.fine_shape
        cz, cx = self.coarse_shape
        full = F.pad(fine.reshape(*fine.shape[:-1], fz-2, fx-2), (1,1,1,1))
        iz, wz = self.axes[0]
        along_z = full.new_zeros((*fine.shape[:-1], cz, fx))
        shape_z = (1,)*len(fine.shape[:-1])+(fz,1)
        indices = iz.reshape(shape_z).expand_as(full)
        along_z.scatter_add_(-2, indices, full*(1-wz[:, None]))
        along_z.scatter_add_(-2, indices+1, full*wz[:, None])
        ix, wx = self.axes[1]
        result = full.new_zeros((*fine.shape[:-1], cz, cx))
        shape_x = (1,)*(fine.ndim)+(fx,)
        indices = ix.reshape(shape_x).expand_as(along_z)
        result.scatter_add_(-1, indices, along_z*(1-wx))
        result.scatter_add_(-1, indices+1, along_z*wx)
        return result[..., 1:-1, 1:-1].reshape(*fine.shape[:-1], -1)

    @property
    def memory_bytes(self):
        return sum(t.numel()*t.element_size() for axis in self.axes for t in axis)


class _Stencil:
    offsets = tuple((z, x) for z in (-1, 0, 1) for x in (-1, 0, 1))

    def __init__(self, padded_shape, action, *, dtype, device):
        self.padded_shape = padded_shape
        self.shape = nz, nx = tuple(n-2 for n in padded_shape)
        self.n_dof = nz*nx
        zz, xx = torch.meshgrid(torch.arange(nz, device=device), torch.arange(nx, device=device), indexing='ij')
        color = (3*(zz % 3)+xx % 3).reshape(-1)
        probes = (torch.arange(9, device=device)[:, None] == color).to(dtype)
        responses = action(probes)
        rows = torch.arange(self.n_dof, device=device)
        coefficients = []
        for dz, dx in self.offsets:
            neighbor_color = (3*((zz+dz) % 3)+(xx+dx) % 3).reshape(-1)
            valid = ((zz+dz >= 0) & (zz+dz < nz) & (xx+dx >= 0) & (xx+dx < nx))
            coefficients.append(responses[neighbor_color, rows].reshape(nz, nx)*valid)
        self.values = torch.stack(coefficients)
        self.diagonal = self.values[4].reshape(-1)

    def apply(self, u, *, adjoint=False):
        nz, nx = self.shape
        full = u.reshape(*u.shape[:-1], nz, nx)
        result = torch.zeros_like(full)
        for coefficient, (dz, dx) in zip(self.values, self.offsets):
            z = slice(max(0, -dz), min(nz, nz-dz))
            x = slice(max(0, -dx), min(nx, nx-dx))
            neighbor_z = slice(max(0, dz), min(nz, nz+dz))
            neighbor_x = slice(max(0, dx), min(nx, nx+dx))
            if adjoint:
                result[..., neighbor_z, neighbor_x] += coefficient[z, x].conj()*full[..., z, x]
            else:
                result[..., z, x] += coefficient[z, x]*full[..., neighbor_z, neighbor_x]
        return result.reshape(u.shape)


class ShiftedHelmholtzMultigrid:
    """One damped-Jacobi V-cycle, rebuilt when model/configuration values change.

    No scalability claim is implied: high frequency or contrast can exceed the
    outer solver budget. Such cases raise SolverFailure instead of downgrading
    accuracy. The cache holds one detached hierarchy on the selected device.
    """

    def __init__(self, operator: HelmholtzOperator, config: MultigridConfig):
        if not isinstance(operator, HelmholtzOperator):
            raise TypeError('geometric multigrid requires the conservative HelmholtzOperator')
        self.operator, self.config = operator, config
        self.clear_cache()

    def clear_cache(self):
        self.levels, self.transfers = [], []
        self._model = self._key = self._lu = self._pivots = None
        self.cache_bytes = self.estimated_peak_bytes = 0

    def _shapes(self):
        shapes = [self.operator.padded_shape]
        while math.prod(n-2 for n in shapes[-1]) > self.config.coarse_max_dof:
            if len(shapes) >= self.config.max_levels:
                break
            coarse = tuple(len(_coarse_nodes(n)) for n in shapes[-1])
            if coarse == shapes[-1]:
                break
            shapes.append(coarse)
        return shapes

    def estimate_memory_bytes(self, m):
        """Conservative explicit tensor estimate; excludes backend allocator pools."""
        shapes = self._shapes()
        size = 16 if m.dtype == torch.float64 else 8
        total = sum(math.prod(shape) for shape in shapes)
        coarse = math.prod(n-2 for n in shapes[-1])
        # Stencils, nine-color setup/transfer temporaries, V-cycle workspace, LU.
        return size*(80*total + 4*coarse**2) + 2*m.numel()*m.element_size()

    @torch.no_grad()
    def prepare(self, m, omega):
        start = time.perf_counter()
        self.operator._validate_model(m)
        if not math.isfinite(float(omega)) or float(omega) <= 0:
            raise ValueError('omega must be finite and strictly positive')
        key = (float(omega), self.operator.grid.config_hash(), self.operator.boundary.config_hash(),
               str(m.dtype), str(m.device), self.config)
        if key == self._key and torch.equal(m, self._model):
            return self
        self.clear_cache()
        shapes = self._shapes()
        self.estimated_peak_bytes = self.estimate_memory_bytes(m)

        def fail(message, reason):
            estimate = self.estimated_peak_bytes
            self.clear_cache()
            raise SolverFailure(message, SolveReport(0., 0., 0, reason, time.perf_counter()-start, estimate, False))
        if math.prod(n-2 for n in shapes[-1]) > self.config.coarse_max_dof:
            fail('max_levels cannot reach the declared coarse_max_dof', 'coarse_grid_limit')
        if self.estimated_peak_bytes > self.config.memory_budget_bytes:
            fail('Multigrid tensor storage exceeds declared memory budget', 'memory_budget_exceeded')
        dtype = torch.complex128 if m.dtype == torch.float64 else torch.complex64
        detached = m.detach()
        mass = self.operator.mass_diagonal(detached, omega)
        fine_action = lambda u: self.operator.apply(detached, omega, u) + 1j*self.config.shift*mass*u
        self.levels.append(_Stencil(shapes[0], fine_action, dtype=dtype, device=m.device))
        for shape in shapes[1:]:
            transfer = _Transfer(self.levels[-1].padded_shape, dtype=m.dtype, device=m.device)
            parent = self.levels[-1]
            action = lambda u: transfer.restrict(parent.apply(transfer.prolong(u)))
            self.levels.append(_Stencil(shape, action, dtype=dtype, device=m.device))
            self.transfers.append(transfer)
        if any(not bool(torch.isfinite(level.values).all()) or bool((level.diagonal.abs() == 0).any()) for level in self.levels):
            fail('Multigrid has a nonfinite or zero smoother diagonal', 'invalid_smoother_diagonal')
        coarse = self.levels[-1]
        dense = coarse.apply(torch.eye(coarse.n_dof, dtype=dtype, device=m.device)).T.contiguous()
        try:
            self._lu, self._pivots = torch.linalg.lu_factor(dense)
        except RuntimeError as exc:
            fail(f'Coarse device factorization failed: {exc}', 'coarse_factorization_failed')
        self._model, self._key = detached.clone(), key
        self.cache_bytes = sum(level.values.numel()*level.values.element_size() for level in self.levels)
        self.cache_bytes += sum(transfer.memory_bytes for transfer in self.transfers)
        self.cache_bytes += sum(t.numel()*t.element_size() for t in (self._model, self._lu, self._pivots))
        return self

    def _cycle(self, level_index, rhs, adjoint):
        if level_index == len(self.levels)-1:
            shape = rhs.shape
            return torch.linalg.lu_solve(self._lu, self._pivots, rhs.reshape(-1, shape[-1]).T,
                                         adjoint=adjoint).T.reshape(shape)
        level = self.levels[level_index]
        diagonal = level.diagonal.conj() if adjoint else level.diagonal
        pre, post = self.config.pre_smooth, self.config.post_smooth
        if adjoint:
            pre, post = post, pre
        value = torch.zeros_like(rhs)
        for _ in range(pre):
            value = value + self.config.damping*(rhs-level.apply(value, adjoint=adjoint))/diagonal
        transfer = self.transfers[level_index]
        residual = rhs-level.apply(value, adjoint=adjoint)
        correction = self._cycle(level_index+1, transfer.restrict(residual), adjoint)
        value = value+transfer.prolong(correction)
        for _ in range(post):
            value = value + self.config.damping*(rhs-level.apply(value, adjoint=adjoint))/diagonal
        return value

    @torch.no_grad()
    def apply(self, m, omega, rhs, *, adjoint=False):
        wanted = torch.complex128 if m.dtype == torch.float64 else torch.complex64
        if rhs.device != m.device or rhs.dtype != wanted:
            raise ValueError('preconditioner RHS dtype/device must match model')
        if rhs.ndim < 1 or rhs.shape[-1] != self.operator.n_dof or not rhs.numel():
            raise ValueError('preconditioner RHS must have nonempty shape [..., n_dof]')
        self.prepare(m, omega)
        # Sequential rows bound standalone V-cycle workspace as well as FGMRES.
        size = rhs.numel()*rhs.element_size()
        if self.estimated_peak_bytes + 2*size > self.config.memory_budget_bytes:
            raise SolverFailure('Multigrid RHS exceeds declared memory budget',
                                SolveReport(0., 0., 0, 'memory_budget_exceeded', 0., self.estimated_peak_bytes+2*size, False))
        result = torch.empty_like(rhs).reshape(-1, self.operator.n_dof)
        for index, row in enumerate(rhs.reshape(-1, self.operator.n_dof)):
            result[index] = self._cycle(0, row, adjoint)
        if not bool(torch.isfinite(result).all()):
            raise SolverFailure('Multigrid produced a nonfinite correction',
                                SolveReport(0., 0., 0, 'nonfinite_preconditioner', 0., self.estimated_peak_bytes, False))
        return result.reshape(rhs.shape)
