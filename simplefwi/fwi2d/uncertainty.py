"""Matrix-free local Gaussian uncertainty in active physical m = 1/v².

The likelihood is 1/2 ||W(d(m)-d_obs)||². W must represent the declared
noise whitening for a statistical interpretation. Only DATA Gauss–Newton
curvature is accepted; a separately supplied proper prior is added once.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
from typing import Callable

import torch


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _norm(v):
    """Scaled norm preserves representable tiny RHS rather than squaring it away."""
    scale = v.abs().max()
    return float(scale * torch.linalg.vector_norm(v/scale)) if scale > 0 else 0.


@dataclass(frozen=True)
class CGConfig:
    rtol: float = 1e-11
    atol: float = 0.
    maxiter: int = 500

    def __post_init__(self):
        for name in ('rtol', 'atol'):
            value = getattr(self, name)
            if not isinstance(value, numbers.Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.rtol == self.atol == 0:
            raise ValueError('at least one convergence tolerance must be positive')
        _integer(self.maxiter, 'maxiter')


@dataclass(frozen=True)
class CGReport:
    converged: bool
    iterations: int
    absolute_residual: float
    relative_residual: float
    convergence_reason: str
    # Conservative Euclidean solution error bound from min(prior diagonal).
    solution_error_bound: float


class PrecisionError(RuntimeError):
    """Improper/indefinite precision or unconverged covariance solve."""
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class SmoothingPrecision:
    """A declared linear factor L and its Euclidean adjoint, giving LᵀL.

    Callers are responsible for the linearity and exact adjoint contract.
    Unlike an arbitrary additive precision action this form is PSD by design.
    """
    forward: Callable
    adjoint: Callable


class PriorPrecision:
    """Proper precision diag(diagonal) + LᵀL, in active physical coordinates.

    The strictly positive finite diagonal is mandatory. No implicit ridge or
    pseudoinverse is introduced. A pure gradient prior is deliberately rejected.
    """
    def __init__(self, diagonal, *, smoothing=None):
        if (not isinstance(diagonal, torch.Tensor) or diagonal.ndim != 1 or
            diagonal.dtype not in (torch.float32, torch.float64) or not diagonal.numel() or
            not bool(torch.isfinite(diagonal).all()) or not bool((diagonal > 0).all())):
            raise ValueError('prior diagonal must be a finite strictly positive real vector')
        if smoothing is not None and not isinstance(smoothing, SmoothingPrecision):
            raise TypeError('smoothing must declare a forward/adjoint SmoothingPrecision factor')
        self.diagonal = diagonal.detach().clone()
        self.smoothing = smoothing

    def _vector(self, v):
        if (not isinstance(v, torch.Tensor) or v.shape != self.diagonal.shape or
            v.dtype != self.diagonal.dtype or v.device != self.diagonal.device or
            not bool(torch.isfinite(v).all())):
            raise ValueError('vector must match active prior shape, real dtype and device and be finite')
        return v.detach()

    def action(self, v):
        v = self._vector(v)
        result = self.diagonal*v
        if self.smoothing is not None:
            result = result + self._vector(self.smoothing.adjoint(self.smoothing.forward(v)))
        return result.detach()


def _cg(action, rhs, prior, config):
    """Diagonal-preconditioned CG, with normalized RHS and true residual checks."""
    rhs = prior._vector(rhs)
    rhs_norm = _norm(rhs)
    lower = float(prior.diagonal.min())
    def report(x, iterations, reason, converged):
        absolute = _norm(rhs-action(x)) if rhs_norm else 0.
        return CGReport(converged, iterations, absolute,
                        absolute/rhs_norm if rhs_norm else 0., reason, absolute/lower)
    x = torch.zeros_like(rhs)
    if rhs_norm == 0:
        return x, report(x, 0, 'zero_rhs', True)
    if rhs_norm <= config.atol:
        return x, report(x, 0, 'absolute_tolerance', True)
    # Normalization avoids under/overflow in the CG inner products for tiny or
    # huge vectors; it does not change the requested unnormalized tolerance.
    b = rhs/rhs_norm
    r = b.clone()
    z = r/prior.diagonal
    p = z.clone()
    rz = torch.dot(r, z)
    target = max(config.rtol, config.atol/rhs_norm)
    for iteration in range(1, config.maxiter+1):
        ap = action(p)
        pap = torch.dot(p, ap)
        if not bool(torch.isfinite(pap)) or pap <= 0 or not bool(torch.isfinite(rz)) or rz <= 0:
            rep = report(x*rhs_norm, iteration-1, 'nonpositive_curvature', False)
            raise PrecisionError('precision is not positive definite on the CG search space', rep)
        alpha = rz/pap
        x = x + alpha*p
        r = r-alpha*ap
        if _norm(r) <= target:
            # Recursive residuals can drift; accept only the true residual.
            true_r = b-action(x)
            if _norm(true_r) <= target:
                result = x*rhs_norm
                rep = report(result, iteration, 'true_residual_tolerance', True)
                if rep.absolute_residual <= max(config.atol, config.rtol*rhs_norm):
                    return result, rep
            r = true_r
            z = r/prior.diagonal
            p = z.clone()
            rz = torch.dot(r, z)
            continue
        z = r/prior.diagonal
        new_rz = torch.dot(r, z)
        p = z+(new_rz/rz)*p
        rz = new_rz
    rep = report(x*rhs_norm, config.maxiter, 'iteration_limit', False)
    raise PrecisionError('covariance/prior CG did not converge to the true residual tolerance', rep)


@dataclass(frozen=True)
class Estimate:
    mean: torch.Tensor
    standard_error: torch.Tensor
    probes: int
    seed: int
    solve_reports: tuple[CGReport, ...]
    max_solve_error_bound: float
    noise_whitening: object = None
    coordinates: str = 'physical_slowness_squared'
    error_description: str = 'pointwise Monte Carlo standard error; inner CG error reported separately'


@dataclass
class Modes:
    """Generalized Ritz modes H u = lambda P u, uᵀ P u = 1.

    Approximate covariance is P⁻¹ - U diag(lambda/(1+lambda)) Uᵀ.
    P⁻¹ is applied iteratively, including optional smoothing. No full covariance
    is stored. Finite-rank corrections omit unresolved data information.
    """
    eigenvalues: torch.Tensor
    vectors: torch.Tensor
    ritz_residual_norms: torch.Tensor
    basis_dimension: int
    exhausted_space: bool
    exact_covariance: bool
    prior_solve_reports: tuple[CGReport, ...]
    _posterior: 'LocalGaussian'
    coordinates: str = 'physical_slowness_squared'

    @property
    def rank(self):
        return self.eigenvalues.numel()

    @property
    def noise_whitening(self):
        return self._posterior.metadata['noise_whitening']

    def covariance_action(self, v):
        v = self._posterior.prior._vector(v)
        base, report = self._posterior._prior_inverse(v)
        self._posterior.last_prior_solve = report
        factors = self.eigenvalues/(1+self.eigenvalues)
        return base-self.vectors@(factors*(self.vectors.T@v))


class LocalGaussian:
    """Data Gauss–Newton likelihood plus one explicit proper prior precision.

    ``curvature.data_gn_hvp`` must be a frozen, symmetric PSD operator on
    [n_active] physical slowness-squared coefficients. Never pass exact Hessian
    actions or theta curvature. Generic actions cannot be certified globally
    without a full spectrum; CG detects encountered nonpositive directions.
    """
    def __init__(self, curvature, prior_precision, *, solve_config=None):
        if not callable(getattr(curvature, 'data_gn_hvp', None)):
            raise TypeError('curvature must expose a data_gn_hvp; exact Hessians are not covariance models')
        coordinates = getattr(curvature, 'coordinates', 'physical_slowness_squared')
        if coordinates not in ('physical_slowness_squared', 'physical_m'):
            raise ValueError('uncertainty requires physical slowness-squared coordinates')
        self.prior = (prior_precision if isinstance(prior_precision, PriorPrecision)
                      else PriorPrecision(prior_precision))
        if getattr(curvature, 'n_active', self.prior.diagonal.numel()) != self.prior.diagonal.numel():
            raise ValueError('prior shape must match curvature active parameters')
        self.curvature = curvature
        self.solve_config = CGConfig() if solve_config is None else solve_config
        if not isinstance(self.solve_config, CGConfig):
            raise TypeError('solve_config must be CGConfig')
        self.last_solve = None
        self.last_prior_solve = None
        whitening = getattr(curvature, 'noise_whitening', None)
        if whitening is None and hasattr(curvature, 'objective'):
            weights = getattr(curvature.objective, 'weights', None)
            whitening = {'convention': 'fixed diagonal W in 1/2 ||W residual||²',
                         'weights': None if weights is None else weights.detach().clone()}
        self.metadata = {'coordinates': 'physical_slowness_squared', 'parameter_units': 's²/m²',
                         'covariance_units': 's⁴/m⁴', 'noise_whitening': whitening,
                         'approximation': 'local data Gauss–Newton Gaussian; explicit prior once',
                         'active_mask': getattr(curvature, 'active_mask', None)}

    def _data_action(self, v):
        result = self.prior._vector(self.curvature.data_gn_hvp(v))
        quadratic = float(torch.dot(v, result))
        roundoff = 100*torch.finfo(v.dtype).eps*_norm(v)*_norm(result)
        if quadratic < -roundoff:
            raise PrecisionError('data_gn_hvp must be positive semidefinite; exact indefinite Hessians are rejected')
        return result

    def precision_action(self, v):
        v = self.prior._vector(v)
        return self._data_action(v)+self.prior.action(v)

    def covariance_action(self, v):
        try:
            value, self.last_solve = _cg(self.precision_action, v, self.prior, self.solve_config)
        except PrecisionError as exc:
            self.last_solve = exc.report
            raise
        return value

    def _prior_inverse(self, v):
        if self.prior.smoothing is None:
            value = self.prior._vector(v)/self.prior.diagonal
            absolute = _norm(v-self.prior.action(value))
            return value, CGReport(True, 0, absolute, absolute/_norm(v) if _norm(v) else 0.,
                                   'diagonal_prior', absolute/float(self.prior.diagonal.min()))
        return _cg(self.prior.action, v, self.prior, self.solve_config)

    def diagonal_estimate(self, probes, seed):
        """Streaming Rademacher Hutchinson diagonal; SE from sample variance.

        Error bars describe probe variation, not local-Gaussian model error.
        At least two probes are required to estimate sampling uncertainty.
        """
        probes = _integer(probes, 'probes', 2)
        seed = _integer(seed, 'seed', 0)
        generator = torch.Generator(device=self.prior.diagonal.device).manual_seed(seed)
        mean = torch.zeros_like(self.prior.diagonal)
        m2 = torch.zeros_like(mean)
        reports = []
        max_error = 0.
        for index in range(1, probes+1):
            z = torch.randint(0, 2, mean.shape, generator=generator, device=mean.device).to(mean.dtype)*2-1
            sample = z*self.covariance_action(z)
            reports.append(self.last_solve)
            max_error = max(max_error, self.last_solve.solution_error_bound)
            delta = sample-mean
            mean = mean+delta/index
            m2 = m2+delta*(sample-mean)
        return Estimate(mean, torch.sqrt(torch.clamp(m2, min=0)/(probes-1)/probes),
                        probes, seed, tuple(reports), max_error,
                        noise_whitening=self.metadata['noise_whitening'])

    def leading_modes(self, rank, seed, *, max_basis=64, max_bytes=64*1024**2):
        """Prior-metric Lanczos with full reorthogonalization and bounded basis.

        Returns largest Ritz values in the explored Krylov space, not a
        certified global spectrum when the basis cap truncates exploration.
        max_basis bounds O(n_active * max_basis) storage; no dense n×n H.
        """
        n = self.prior.diagonal.numel()
        rank = _integer(rank, 'rank')
        max_basis = _integer(max_basis, 'max_basis')
        max_bytes = _integer(max_bytes, 'max_bytes')
        seed = _integer(seed, 'seed', 0)
        if rank > n or rank > max_basis:
            raise ValueError('rank must not exceed n_active or max_basis')
        dimension = min(n, max_basis, max(2*rank+8, 20))
        # Q, PQ, HQ, candidate work and small projected eigensystem.
        required = (9*n*dimension+16*n+8*dimension**2)*self.prior.diagonal.element_size()
        if required > max_bytes:
            raise MemoryError(f'Lanczos workspace requires approximately {required} bytes > max_bytes')
        generator = torch.Generator(device=self.prior.diagonal.device).manual_seed(seed)
        candidate = torch.randn(n, generator=generator, dtype=self.prior.diagonal.dtype,
                                device=self.prior.diagonal.device)
        basis, p_basis, h_basis, reports = [], [], [], []
        for index in range(dimension):
            # A single Krylov chain cannot span repeated-eigenvalue spaces.
            # Restart orthogonally after invariant-subspace breakdown, rather
            # than silently claiming a requested full-rank result is complete.
            for attempt in range(3):
                pc = self.prior.action(candidate)
                original_norm = math.sqrt(max(float(candidate@pc), 0.))
                for _ in range(2):
                    for q, pq in zip(basis, p_basis):
                        coefficient = q@pc
                        candidate = candidate-coefficient*q
                        pc = pc-coefficient*pq
                norm = math.sqrt(max(float(candidate@pc), 0.))
                if norm > 100*torch.finfo(candidate.dtype).eps*original_norm:
                    break
                candidate = torch.randn(n, generator=generator, dtype=candidate.dtype,
                                        device=candidate.device)
            else:
                raise PrecisionError('could not extend a numerically independent prior-metric basis')
            q, pq = candidate/norm, pc/norm
            hq = self._data_action(q)
            basis.append(q); p_basis.append(pq); h_basis.append(hq)
            if index+1 < dimension:
                candidate, report = self._prior_inverse(hq)
                reports.append(report)
        q = torch.stack(basis, dim=1)
        pq = torch.stack(p_basis, dim=1)
        hq = torch.stack(h_basis, dim=1)
        projected = q.T@hq
        if not torch.allclose(projected, projected.T, rtol=1e-7, atol=100*torch.finfo(q.dtype).eps*float(projected.abs().max())):
            raise PrecisionError('data/prior actions violate symmetry in the Lanczos metric')
        values, rotation = torch.linalg.eigh((projected+projected.T)/2)
        tolerance = 1000*torch.finfo(q.dtype).eps*max(float(values.abs().max()), 1e-300)
        if float(values.min()) < -tolerance:
            raise PrecisionError('data Gauss–Newton curvature must be positive semidefinite')
        selected = min(rank, values.numel())
        values = values.flip(0)[:selected].clamp_min(0)
        rotation = rotation.flip(1)[:, :selected]
        vectors = q@rotation
        residual = hq@rotation-(pq@rotation)*values
        residuals = torch.tensor([_norm(residual[:, i]) for i in range(selected)],
                                 dtype=q.dtype, device=q.device)
        exhausted = q.shape[1] == n
        return Modes(values, vectors, residuals, q.shape[1], exhausted,
                     exhausted and selected == n, tuple(reports), self)
