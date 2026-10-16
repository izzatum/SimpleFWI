"""Explicit CPU SciPy reference solves with value-verified bounded LU reuse.

This detached adapter is the numerical primitive for implicit differentiation;
its NumPy conversion intentionally does not construct a PyTorch autograd graph.
The memory budget bounds retained factor state, not transient factorization
workspace (whose backend-dependent allocation is only estimated).
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
import torch

from ..types import SolveReport, SolverConfig
from .base import SolverFailure


@dataclass
class _Factor:
    model: torch.Tensor
    lu: object
    memory_bytes: int


def _sparse_bytes(matrix):
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes


def _norm_rows(values):
    """Scaled Euclidean norms avoid squaring tiny/large RHS into under/overflow."""
    magnitudes = np.abs(values)
    scale = magnitudes.max(axis=-1)
    safe_scale = np.where(scale == 0, 1., scale)
    return scale * np.sqrt(np.sum((magnitudes / safe_scale[:, None])**2, axis=-1))


class DirectSolver:
    """Reusable CPU CSC factorization with checked true residuals.

    Use :meth:`evaluation` around an objective/closure evaluation to release
    factors promptly, or :meth:`clear_cache` at explicit stage boundaries.
    Numeric state is always verified against detached model values, even when
    a caller omits these lifecycle helpers or mutates a tensor in place.
    """

    def __init__(self, config: SolverConfig | None = None):
        self.config = SolverConfig() if config is None else config
        if self.config.method != 'direct':
            raise ValueError('DirectSolver requires method=direct')
        if torch.device(self.config.device).type != 'cpu':
            raise ValueError('DirectSolver is a CPU adapter; select device=cpu explicitly')
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self.factorization_count = 0

    @property
    def cache_bytes(self):
        return self._cache_bytes

    @property
    def cache_entries(self):
        return len(self._cache)

    def clear_cache(self):
        """Release all retained numeric factors and detached model snapshots."""
        self._cache.clear()
        self._cache_bytes = 0

    @contextmanager
    def evaluation(self):
        """Bound numeric-state lifetime to one model evaluation, including errors."""
        self.clear_cache()
        try:
            yield self
        finally:
            self.clear_cache()

    def _key(self, operator, m, omega):
        snapshot = m.detach().resolve_conj().resolve_neg().contiguous()
        digest = hashlib.sha256(snapshot.numpy().tobytes()).digest()
        settings = tuple(getattr(operator, name).config_hash()
                         for name in ('grid', 'boundary'))
        key = (id(operator), type(operator), settings, float(omega), tuple(m.shape),
               str(m.dtype), str(m.device), self.config.config_hash(), digest)
        return key, snapshot

    def _report(self, start, relative=0., absolute=0., memory=0,
                reason='direct_factorization', converged=True):
        # A failure can have nonfinite raw diagnostics; keep report serializable.
        limit = np.finfo(np.float64).max
        relative = float(np.nan_to_num(relative, nan=limit, posinf=limit, neginf=limit))
        absolute = float(np.nan_to_num(absolute, nan=limit, posinf=limit, neginf=limit))
        return SolveReport(relative, absolute, 1, reason, time.perf_counter()-start,
                           int(memory), converged)

    def solve(self, operator, m: torch.Tensor, omega: float, rhs: torch.Tensor,
              *, adjoint: bool = False) -> tuple[torch.Tensor, SolveReport]:
        """Solve A u=q or A^H u=q; source/batch axes precede the compact axis."""
        start = time.perf_counter()
        if not math.isfinite(float(omega)) or float(omega) <= 0:
            raise ValueError('omega must be finite and strictly positive')
        if not isinstance(m, torch.Tensor) or not isinstance(rhs, torch.Tensor):
            raise TypeError('m and rhs must be torch tensors')
        if m.device.type != 'cpu' or rhs.device.type != 'cpu':
            raise ValueError('DirectSolver requires CPU tensors; transfer devices explicitly')
        real_dtype = torch.float64 if self.config.dtype == torch.complex128 else torch.float32
        if m.dtype != real_dtype or rhs.dtype != self.config.dtype:
            raise ValueError('model and RHS dtype must match declared precision')
        if tuple(m.shape) != operator.grid.shape:
            raise ValueError('model shape must equal the physical grid shape')
        if rhs.ndim < 1 or rhs.shape[-1] != operator.n_dof or rhs.numel() == 0:
            raise ValueError('rhs must have nonempty shape [..., n_dof]')
        if not bool(torch.isfinite(m).all()) or not bool((m > 0).all()):
            raise ValueError('m must be finite and positive')
        if not bool(torch.isfinite(rhs).all()):
            raise ValueError('rhs must be finite')
        key, snapshot = self._key(operator, m, omega)
        factor = self._cache.get(key)
        if factor is not None and not torch.equal(snapshot, factor.model):
            # Collision protection: actual model values decide numeric identity.
            self._cache_bytes -= self._cache.pop(key).memory_bytes
            factor = None
        assembly_bytes = 0
        if factor is None:
            dtype = np.complex128 if self.config.dtype == torch.complex128 else np.complex64
            matrix = sp.csc_matrix(operator.assemble(m.detach(), float(omega)), dtype=dtype)
            if matrix.shape != (operator.n_dof, operator.n_dof):
                raise ValueError('assembled operator shape does not match n_dof')
            assembly_bytes = _sparse_bytes(matrix)
            try:
                lu = splu(matrix)
            except (RuntimeError, ValueError) as exc:
                report = self._report(start, memory=assembly_bytes,
                                      reason='factorization_failed', converged=False)
                raise SolverFailure(f'CPU sparse factorization failed: {exc}', report) from exc
            self.factorization_count += 1
            factor_bytes = (_sparse_bytes(lu.L) + _sparse_bytes(lu.U)
                            + lu.perm_r.nbytes + lu.perm_c.nbytes
                            + snapshot.numel()*snapshot.element_size())
            factor = _Factor(snapshot.clone(), lu, factor_bytes)
            budget = self.config.memory_budget_bytes
            if factor_bytes <= budget:
                while self._cache and self._cache_bytes + factor_bytes > budget:
                    _, removed = self._cache.popitem(last=False)
                    self._cache_bytes -= removed.memory_bytes
                self._cache[key] = factor
                self._cache_bytes += factor_bytes
        else:
            self._cache.move_to_end(key)
        q = rhs.detach().resolve_conj().resolve_neg().contiguous().numpy().reshape(-1, operator.n_dof)
        memory = self._cache_bytes + assembly_bytes + 4*q.nbytes
        if key not in self._cache:
            memory += factor.memory_bytes
        try:
            # SciPy expects [n_dof, nrhs]; 'H' includes complex conjugation.
            solved = factor.lu.solve(np.asfortranarray(q.T), trans='H' if adjoint else 'N')
        except (RuntimeError, ValueError) as exc:
            raise SolverFailure('CPU sparse solve failed', self._report(
                start, memory=memory, reason='solve_failed', converged=False)) from exc
        result = torch.from_numpy(np.array(solved.T, copy=True)).reshape(rhs.shape)
        with torch.no_grad():
            residual = operator.apply(m.detach(), float(omega), result, adjoint=adjoint) - rhs.detach()
        rnorm = _norm_rows(residual.detach().resolve_conj().resolve_neg().numpy().reshape(-1, operator.n_dof))
        qnorm = _norm_rows(q)
        relative = np.divide(rnorm, qnorm, out=np.zeros_like(rnorm), where=qnorm != 0)
        # Each RHS must pass; an easy source cannot hide another source's error.
        accepted = np.isfinite(rnorm) & ((rnorm <= self.config.atol)
                                       | ((qnorm > 0) & (relative <= self.config.rtol)))
        converged = bool(np.all(accepted)) and bool(torch.isfinite(result).all())
        report = self._report(start, relative.max(), rnorm.max(), memory,
                              'direct_factorization' if converged else 'residual_tolerance_failed',
                              converged)
        if not converged:
            raise SolverFailure('CPU direct solve failed true residual tolerance', report)
        return result, report
