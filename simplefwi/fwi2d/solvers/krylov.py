"""Device-resident restarted right FGMRES with checked physical residuals.

RHS rows are solved sequentially so the Arnoldi storage is independent of batch
size. The returned solution necessarily scales with the supplied batch. Only
scalar stopping decisions synchronize a CUDA device; no wavefield or numeric
factor is transferred to CPU. Use implicit_solve for differentiable solves.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import time

import torch

from ..types import SolveReport, SolverConfig
from .base import SolverFailure


def _norm(x):
    """Scaled FP64 norm, including very small/large finite RHS values."""
    magnitude = x.abs().to(torch.float64)
    scale = magnitude.amax()
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return scale * torch.linalg.vector_norm(magnitude / safe)


def _finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else torch.finfo(torch.float64).max


class FGMRESSolver:
    """Flexible GMRES; config.maxiter applies independently to every RHS.

    ``preconditioner.apply(m, omega, rhs, adjoint=...)`` approximates the inverse
    on the right and may change between iterations. None is the diagnostic
    unpreconditioned baseline. Reports count the maximum iterations over RHS.
    Memory reports are conservative tensor-storage estimates, not allocator RSS.
    """

    def __init__(self, config: SolverConfig, preconditioner=None):
        if config.method != 'fgmres':
            raise ValueError('FGMRESSolver requires method=fgmres')
        self.config = config
        self.preconditioner = preconditioner
        self.basis_peak_vectors = 0

    @property
    def cache_bytes(self):
        return getattr(self.preconditioner, 'cache_bytes', 0)

    def clear_cache(self):
        if hasattr(self.preconditioner, 'clear_cache'):
            self.preconditioner.clear_cache()

    @contextmanager
    def evaluation(self):
        self.clear_cache()
        try:
            yield self
        finally:
            self.clear_cache()

    def estimate_memory_bytes(self, n_dof, rhs_count=1):
        """Declared upper estimate, including returned batch and MG workspace."""
        k = min(self.config.restart, self.config.maxiter, n_dof)
        size = torch.empty((), dtype=self.config.dtype).element_size()
        # V[k+1,n], Z[k,n], residual/solution/operator temporaries and small QR.
        own = size*((2*k+24)*n_dof + 2*rhs_count*n_dof + 4*(k+1)**2)
        return own + getattr(self.preconditioner, 'estimated_peak_bytes', self.cache_bytes)

    @torch.no_grad()
    def solve(self, operator, m, omega, rhs, *, adjoint=False):
        start = time.perf_counter()
        if not math.isfinite(float(omega)) or float(omega) <= 0:
            raise ValueError('omega must be finite and strictly positive')
        if not isinstance(m, torch.Tensor) or not isinstance(rhs, torch.Tensor):
            raise TypeError('m and rhs must be torch tensors')
        declared = torch.device(self.config.device)
        for tensor in (m, rhs):
            if tensor.device.type != declared.type or (declared.index is not None and tensor.device.index != declared.index):
                raise ValueError('model and RHS device must match declared device')
        if m.device != rhs.device:
            raise ValueError('model and RHS must share a device')
        real_dtype = torch.float64 if self.config.dtype == torch.complex128 else torch.float32
        if m.dtype != real_dtype or rhs.dtype != self.config.dtype:
            raise ValueError('model and RHS dtype must match declared precision')
        if tuple(m.shape) != operator.grid.shape:
            raise ValueError('model shape must equal physical grid shape')
        if rhs.ndim < 1 or rhs.shape[-1] != operator.n_dof or not rhs.numel():
            raise ValueError('rhs must have nonempty shape [..., n_dof]')
        if not bool(torch.isfinite(m).all() & (m > 0).all()) or not bool(torch.isfinite(rhs).all()):
            raise ValueError('model must be finite positive and RHS finite')
        m, rhs = m.detach(), rhs.detach()
        n, count = operator.n_dof, rhs.numel()//operator.n_dof
        self.basis_peak_vectors = 0
        memory, iterations = self.estimate_memory_bytes(n, count), 0
        if hasattr(self.preconditioner, 'estimate_memory_bytes'):
            current_estimate = getattr(self.preconditioner, 'estimated_peak_bytes', self.cache_bytes)
            memory += self.preconditioner.estimate_memory_bytes(m)-current_estimate
        max_relative, max_absolute = 0., 0.

        def report(reason, converged, relative=None, absolute=None):
            if rhs.device.type == 'cuda':
                torch.cuda.synchronize(rhs.device)
            return SolveReport(_finite_number(max_relative if relative is None else relative),
                               _finite_number(max_absolute if absolute is None else absolute),
                               iterations, reason, time.perf_counter()-start, memory, converged)

        def guard():
            if memory > self.config.memory_budget_bytes:
                raise SolverFailure('FGMRES tensor storage exceeds declared memory budget',
                                    report('memory_budget_exceeded', False))
        guard()
        if hasattr(self.preconditioner, 'operator') and self.preconditioner.operator is not operator:
            raise ValueError('preconditioner must be bound to the solve operator')
        if hasattr(self.preconditioner, 'prepare'):
            self.preconditioner.prepare(m, float(omega))
            memory = self.estimate_memory_bytes(n, count)
            guard()
        result = torch.empty_like(rhs).reshape(count, n)
        restart = min(self.config.restart, self.config.maxiter, n)
        eps = torch.finfo(real_dtype).eps
        apply = lambda x: operator.apply(m, float(omega), x, adjoint=adjoint)
        for row_index, b in enumerate(rhs.reshape(count, n)):
            x = torch.zeros_like(b)
            bnorm = _norm(b)
            if not bool(torch.isfinite(bnorm)):
                raise SolverFailure('FGMRES RHS norm is not representable in float64',
                                    report('nonfinite_rhs_norm', False, bnorm, bnorm))
            threshold = torch.maximum(bnorm*self.config.rtol, bnorm.new_tensor(self.config.atol))
            residual, rnorm = b.clone(), bnorm
            row_iterations = 0
            while bool(rnorm > threshold):
                k = min(restart, self.config.maxiter-row_iterations)
                if k == 0:
                    raise SolverFailure('FGMRES exhausted maxiter before true residual convergence',
                                        report('maxiter_exhausted', False, rnorm/bnorm, rnorm))
                v = torch.empty((k+1, n), dtype=rhs.dtype, device=rhs.device)
                z = torch.empty((k, n), dtype=rhs.dtype, device=rhs.device)
                h = torch.zeros((k+1, k), dtype=rhs.dtype, device=rhs.device)
                g = torch.zeros(k+1, dtype=rhs.dtype, device=rhs.device)
                cs = torch.empty(k, dtype=real_dtype, device=rhs.device)
                sn = torch.empty(k, dtype=rhs.dtype, device=rhs.device)
                self.basis_peak_vectors = max(self.basis_peak_vectors, 2*k+1)
                v[0], g[0] = residual/rnorm, rnorm
                for j in range(k):
                    zj = (v[j] if self.preconditioner is None else
                          self.preconditioner.apply(m, float(omega), v[j], adjoint=adjoint))
                    if zj.shape != b.shape or zj.dtype != b.dtype or zj.device != b.device:
                        raise ValueError('preconditioner must preserve RHS shape, dtype and device')
                    z[j] = zj
                    del zj  # a baseline V-row view must not retain the old basis
                    w = apply(z[j])
                    original_norm = _norm(w)
                    # Two MGS passes suppress loss of orthogonality on indefinite
                    # and nonnormal systems; Z retains every flexible correction.
                    for _ in range(2):
                        for i in range(j+1):
                            coefficient = torch.sum(v[i].conj()*w, dtype=torch.complex128).to(rhs.dtype)
                            h[i, j] += coefficient
                            w = w-coefficient*v[i]
                    next_norm = _norm(w)
                    h[j+1, j] = next_norm
                    breakdown = bool(next_norm <= 10*eps*original_norm)
                    if not breakdown:
                        v[j+1] = w/next_norm
                    for i in range(j):
                        a, bb = h[i, j].clone(), h[i+1, j].clone()
                        h[i, j] = cs[i]*a+sn[i]*bb
                        h[i+1, j] = -sn[i].conj()*a+cs[i]*bb
                    a, bb = h[j, j].clone(), h[j+1, j].clone()
                    length = torch.hypot(a.abs(), bb.abs())
                    if not bool(torch.isfinite(length)) or bool(length == 0):
                        raise SolverFailure('FGMRES Arnoldi breakdown without convergence',
                                            report('arnoldi_breakdown', False, rnorm/bnorm, rnorm))
                    phase = torch.where(a.abs() > 0, a/torch.where(a.abs() > 0, a.abs(), torch.ones_like(a.abs())), torch.ones_like(a))
                    cs[j], sn[j] = a.abs()/length, phase*bb.conj()/length
                    h[j, j], h[j+1, j] = phase*length, 0
                    old_g = g[j].clone()
                    g[j], g[j+1] = cs[j]*old_g, -sn[j].conj()*old_g
                    row_iterations += 1
                    iterations = max(iterations, row_iterations)
                    # Never accept the QR estimate without the physical A residual.
                    if bool(g[j+1].abs() <= threshold) or j == k-1 or breakdown:
                        try:
                            y = torch.linalg.solve_triangular(h[:j+1, :j+1], g[:j+1, None], upper=True)[:, 0]
                        except RuntimeError as exc:
                            raise SolverFailure('FGMRES triangular solve failed', report('arnoldi_breakdown', False)) from exc
                        candidate = x + y @ z[:j+1]
                        checked = b-apply(candidate)
                        checked_norm = _norm(checked)
                        if not bool(torch.isfinite(checked_norm)):
                            raise SolverFailure('FGMRES produced a nonfinite residual', report('nonfinite_residual', False, checked_norm, checked_norm))
                        if bool(checked_norm <= threshold) or j == k-1 or breakdown:
                            x, residual, rnorm = candidate, checked, checked_norm
                            break
                # Buffers are freed before the next restart; no iteration graph.
                del v, z, h, g, cs, sn
            result[row_index] = x
            max_absolute = max(max_absolute, float(rnorm))
            relative = float(rnorm/bnorm) if bool(bnorm > 0) else 0.
            max_relative = max(max_relative, relative)
        return result.reshape(rhs.shape), report('true_residual_tolerance', True)
