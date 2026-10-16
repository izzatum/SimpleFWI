"""Source-streamed Jacobian and real Hessian actions; no default J/H assembly.

Working wavefield/autograd storage scales with ``source_batch_size`` (one by
 default), not source count. Returned JVP data necessarily has the full data
shape. Solver factor storage is governed separately by its configured budget.
Actions are numerical operators at a frozen reference, not differentiable
building blocks for third derivatives.
"""
from __future__ import annotations

import copy
import math
import numbers
import torch

from .autograd import implicit_solve


class Curvature:
    """Actions in physical slowness squared m, with real [n_active] vectors.

    J is the unweighted complex data Jacobian. VJP means Re(J^H y), and
    data GN is Re(J^H W^H W J). ``gn_hvp`` adds the physical regularizer;
    ``exact_hvp`` also includes residual-dependent data curvature. The model
    and mask are cloned; recreate this instance after changing the reference.
    The objective configuration (geometry, observations, weights) stays fixed
    by caller contract.
    """
    coordinates = 'physical_slowness_squared'

    def __init__(self, objective, m, *, active_mask=None, source_batch_size=1):
        objective._validate_model(m)
        self.objective = objective
        self.source_batch_size = objective._batch_size(source_batch_size)
        self._m = m.detach().clone()
        mask = torch.ones_like(m, dtype=torch.bool) if active_mask is None else active_mask
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or
                mask.shape != m.shape or mask.device != m.device or not bool(mask.any())):
            raise ValueError('active_mask must be a nonempty boolean mask matching model shape/device')
        self._active_mask = mask.detach().clone()
        self.n_active = int(mask.sum())
        self.dtype, self.device = m.dtype, m.device

    @property
    def m(self):
        return self._m.clone()

    @property
    def reference_model(self):
        """A copy of the frozen physical reference model."""
        return self._m.clone()

    @property
    def active_mask(self):
        return self._active_mask.clone()

    def _vector(self, vector):
        if (not isinstance(vector, torch.Tensor) or vector.shape != (self.n_active,) or
                vector.dtype != self.dtype or vector.device != self.device or
                not bool(torch.isfinite(vector).all())):
            raise ValueError('model vector must be finite real [n_active] with reference dtype/device')
        return vector.detach()

    def inject(self, vector):
        """Inject an active real vector into a zero full physical grid."""
        return torch.zeros_like(self._m).masked_scatter(self._active_mask, self._vector(vector))

    def restrict(self, grid_vector):
        """Restrict a full physical grid to the frozen active-cell ordering."""
        if (grid_vector.shape != self._m.shape or grid_vector.dtype != self.dtype or
                grid_vector.device != self.device):
            raise ValueError('grid vector must match reference model shape/dtype/device')
        return grid_vector[self._active_mask].detach()

    def _batches(self):
        ns = len(self.objective.acquisition.sources_m)
        for fi, frequency in enumerate(self.objective.frequencies_hz):
            for start in range(0, ns, self.source_batch_size):
                yield fi, start, min(start+self.source_batch_size, ns), 2*math.pi*frequency

    def _wave(self, m, start, stop, omega):
        o = self.objective
        return implicit_solve(m, o.source_rhs(m, start, stop), omega, o.operator, o.solver)

    def _increment(self, dm, wave, omega):
        o = self.objective
        rhs = -o.operator.model_derivative(dm, omega, wave)
        return implicit_solve(self._m, rhs, omega, o.operator, o.solver)

    def _pullback(self, wave, data, omega):
        o = self.objective
        with torch.no_grad():
            dual = implicit_solve(self._m, o.receivers.scatter(data), omega,
                                  o.operator, o.solver, adjoint=True)
        # Only the local linear stencil product has a graph; no wavefield graph
        # survives this contraction. This includes nearest-boundary PML scatter.
        with torch.enable_grad():
            dm = torch.zeros_like(self._m, requires_grad=True)
            product = o.operator.model_derivative(dm, omega, wave.detach())
            return torch.autograd.grad(product, dm, grad_outputs=-dual)[0].detach()

    def jvp(self, vector):
        dm = self.inject(vector)
        result = torch.empty_like(self.objective.observed)
        with torch.no_grad():
            for fi, start, stop, omega in self._batches():
                wave = self._wave(self._m, start, stop, omega)
                increment = self._increment(dm, wave, omega)
                result[fi, start:stop] = self.objective.receivers.sample(increment)
        return result

    def vjp(self, data_vector):
        o = self.objective
        if (not isinstance(data_vector, torch.Tensor) or data_vector.shape != o.observed.shape or
                data_vector.dtype != o.observed.dtype or data_vector.device != self.device or
                not bool(torch.isfinite(data_vector).all())):
            raise ValueError('data vector must be finite and match observed shape/dtype/device')
        result = torch.zeros_like(self._m)
        for fi, start, stop, omega in self._batches():
            with torch.no_grad():
                wave = self._wave(self._m, start, stop, omega)
            result += self._pullback(wave, data_vector[fi, start:stop].detach(), omega)
        return self.restrict(result)

    def data_gn_hvp(self, vector):
        dm = self.inject(vector)
        o = self.objective
        result = torch.zeros_like(self._m)
        for fi, start, stop, omega in self._batches():
            with torch.no_grad():
                wave = self._wave(self._m, start, stop, omega)
                data = o.receivers.sample(self._increment(dm, wave, omega))
                weights = o.weights[fi, start:stop]
                # Follow the loss's promoted W*data arithmetic before applying
                # W^H. Squaring weights first can under/overflow their lower
                # precision. Autograd casts the final cotangent back to the
                # prediction dtype; the adjoint PDE requires that same dtype.
                weighted = (weights.conj()*(weights*data)).to(data.dtype)
            result += self._pullback(wave, weighted, omega)
        return self.restrict(result)

    def _regularizer_hvp(self, dm):
        if self.objective.regularizer is None:
            return torch.zeros_like(self._m)
        with torch.enable_grad():
            m = self._m.clone().requires_grad_()
            gradient, = torch.autograd.grad(self.objective.regularizer.value(m), m, create_graph=True)
            return torch.autograd.grad(gradient, m, grad_outputs=dm)[0].detach()

    def gn_hvp(self, vector):
        return self.data_gn_hvp(vector)+self.restrict(self._regularizer_hvp(self.inject(vector)))

    def exact_hvp(self, vector):
        dm = self.inject(vector)
        result = torch.zeros_like(self._m)
        # Consume the second-order graph before requesting the next source batch.
        with torch.enable_grad():
            m = self._m.clone().requires_grad_()
            for loss in self.objective.iter_data_losses(m, self.source_batch_size):
                gradient, = torch.autograd.grad(loss, m, create_graph=True)
                result += torch.autograd.grad(gradient, m, grad_outputs=dm)[0].detach()
                del loss, gradient
        return self.restrict(result+self._regularizer_hvp(dm))


class TransformedCurvature:
    """Actions in active dense theta coordinates at a frozen parameter model.

    Exact theta Hessians differentiate each loss through the full nonlinear
    transformation, including its second derivative. GN uses D^T H_GN,m D;
    the physical regularizer's exact transformed Hessian is added once by
    ``gn_hvp``. Sparse embeddings can use :class:`Curvature` in physical m;
    no native sparse double-backward support is assumed here.
    """
    coordinates = 'theta'

    def __init__(self, objective, model, *, source_batch_size=1):
        if model.representation != 'dense':
            raise ValueError('theta curvature requires the dense representation; use physical Curvature for sparse models')
        self._model = copy.deepcopy(model)
        self._theta = self._model.raw_parameter.detach().clone()
        self._mask = self._model.active_mask.detach().clone()
        self._physical = Curvature(objective, self._model.slowness_squared(),
                                   active_mask=self._mask, source_batch_size=source_batch_size)
        self.objective = objective
        self.source_batch_size = self._physical.source_batch_size
        self.n_active = self._physical.n_active
        self.dtype, self.device = self._theta.dtype, self._theta.device

    def _map(self, active):
        return self._model.slowness_squared_from(self._theta.masked_scatter(self._mask, active))

    def _push(self, vector):
        vector = self._physical._vector(vector)
        with torch.enable_grad():
            _, dm = torch.autograd.functional.jvp(self._map, self._theta[self._mask], vector)
        return self._physical.restrict(dm)

    def _pull(self, physical_vector):
        with torch.enable_grad():
            active = self._theta[self._mask].clone().requires_grad_()
            m = self._map(active)
            return torch.autograd.grad(m, active, grad_outputs=self._physical.inject(physical_vector))[0].detach()

    def jvp(self, vector):
        return self._physical.jvp(self._push(vector))

    def vjp(self, data_vector):
        return self._pull(self._physical.vjp(data_vector))

    def data_gn_hvp(self, vector):
        return self._pull(self._physical.data_gn_hvp(self._push(vector)))

    def _hvp(self, loss, active, vector):
        gradient, = torch.autograd.grad(loss, active, create_graph=True)
        return torch.autograd.grad(gradient, active, grad_outputs=vector)[0].detach()

    def _regularizer_hvp(self, vector):
        if self.objective.regularizer is None:
            return torch.zeros_like(vector)
        with torch.enable_grad():
            active = self._theta[self._mask].clone().requires_grad_()
            return self._hvp(self.objective.regularizer.value(self._map(active)), active, vector)

    def gn_hvp(self, vector):
        vector = self._physical._vector(vector)
        return self.data_gn_hvp(vector)+self._regularizer_hvp(vector)

    def exact_hvp(self, vector):
        vector = self._physical._vector(vector)
        result = torch.zeros_like(vector)
        o = self.objective
        with torch.enable_grad():
            for fi, start, stop, omega in self._physical._batches():
                # A fresh transform graph per batch prevents graph retention
                # across all source solves and preserves its second derivative.
                active = self._theta[self._mask].clone().requires_grad_()
                m = self._map(active)
                wave = self._physical._wave(m, start, stop, omega)
                residual = o.weights[fi, start:stop]*(o.receivers.sample(wave)-o.observed[fi, start:stop])
                loss = 0.5*(residual.real.square()+residual.imag.square()).sum()
                result += self._hvp(loss, active, vector)
                del loss, residual, wave, m, active
        return result+self._regularizer_hvp(vector)


def explicit_matrix(action, n, *, max_bytes):
    """Build a tiny real diagnostic matrix, guarding bytes before allocation.

    Budget includes the result and two n-vectors (input/output); internal action
    workspace and solver factors are separate. Bound actions use their owner's
    dtype/device; standalone actions use CPU float64. No n-by-n identity or
    retained list of columns is allocated.
    """
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1:
        raise ValueError('n must be a positive integer')
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, numbers.Integral) or max_bytes < 0:
        raise ValueError('max_bytes must be a nonnegative integer')
    owner = getattr(action, '__self__', None)
    dtype, device = getattr(owner, 'dtype', torch.float64), getattr(owner, 'device', 'cpu')
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('explicit diagnostic matrix requires real float32 or float64')
    item_bytes = 8 if dtype == torch.float64 else 4
    required = (n*n+2*n)*item_bytes
    if required > max_bytes:
        raise MemoryError(f'explicit matrix requires {required} bytes, exceeding max_bytes={max_bytes}')
    matrix = torch.empty((n, n), dtype=dtype, device=device)
    direction = torch.zeros(n, dtype=dtype, device=device)
    for j in range(n):
        direction.zero_(); direction[j] = 1
        column = action(direction)
        if column.shape != (n,) or column.dtype != dtype or column.device != direction.device:
            raise ValueError('action output must match diagnostic vector shape/dtype/device')
        matrix[:, j] = column.detach()
        del column
    return matrix
