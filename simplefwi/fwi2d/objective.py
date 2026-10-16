"""Weighted data misfit and physical regularization with explicit summation."""
from __future__ import annotations

import math
import numbers
import numpy as np
import torch

from .acquisition import SamplingOperator
from .autograd import implicit_solve


class FWIObjective:
    """One-half ||W(predicted-observed)||² + one physical regularizer.

    Data axes are [frequency,source,receiver]. W is a fixed complex or real
    scalar/tensor broadcastable to those axes; no data-count normalization is
    applied. Observations and weights are frozen copies. ``iter_data_losses``
    excludes regularization so a streaming backward adds it exactly once.
    Python weight scalars/sequences use the observations' complex precision;
    explicitly typed tensor/NumPy weights retain their declared precision.
    """
    def __init__(self, operator, acquisition, frequencies_hz, observed, weights,
                 regularizer, solver):
        self.operator, self.acquisition = operator, acquisition
        self.frequencies_hz = tuple(float(f) for f in frequencies_hz)
        if not self.frequencies_hz or any(not math.isfinite(f) or f <= 0
                                         for f in self.frequencies_hz):
            raise ValueError('frequencies_hz must be nonempty, finite, and positive')
        shape = (len(self.frequencies_hz), len(acquisition.sources_m),
                 len(acquisition.receivers_m))
        if not isinstance(observed, torch.Tensor) or tuple(observed.shape) != shape:
            raise ValueError(f'observed must have [frequency,source,receiver] shape {shape}')
        if observed.dtype not in (torch.complex64, torch.complex128):
            raise ValueError('observed must explicitly use complex64 or complex128')
        if not bool(torch.isfinite(observed).all()):
            raise ValueError('observed data must be finite')
        self.observed = observed.detach().clone()
        if isinstance(weights, torch.Tensor) and weights.device != observed.device:
            raise ValueError('weights and observations must share an explicit device')
        if isinstance(weights, (torch.Tensor, np.ndarray, np.generic)):
            # Preserve explicitly chosen input precision, even if it is higher
            # than the wavefield precision. PyTorch promotes the products.
            weights = torch.as_tensor(weights, device=observed.device)
        else:
            # Do not round Python float/complex values through the default
            # float32/complex64 tensor dtype before constructing the objective.
            weights = torch.as_tensor(weights, dtype=observed.dtype, device=observed.device)
        if not bool(torch.isfinite(weights).all()):
            raise ValueError('weights must be finite')
        try:
            self.weights = torch.broadcast_to(weights, shape).detach().clone()
        except RuntimeError as exc:
            raise ValueError('weights must broadcast to [frequency,source,receiver]') from exc
        self.regularizer, self.solver = regularizer, solver
        self.receivers = SamplingOperator(operator.grid, acquisition.receivers_m,
                                          boundary=operator.boundary)
        # Validate every source before the first solve, even with small batches.
        self.sources = SamplingOperator(operator.grid, acquisition.sources_m,
                                        boundary=operator.boundary)

    def _batch_size(self, source_batch_size):
        if source_batch_size is None:
            return len(self.acquisition.sources_m)
        if (isinstance(source_batch_size, bool) or
            not isinstance(source_batch_size, numbers.Integral) or source_batch_size < 1):
            raise ValueError('source_batch_size must be a positive integer')
        return int(source_batch_size)

    def _validate_model(self, m):
        self.operator._validate_model(m)
        dtype = torch.complex128 if m.dtype == torch.float64 else torch.complex64
        if m.device != self.observed.device or dtype != self.observed.dtype:
            raise ValueError('model and observations must share explicit device and precision')

    def source_rhs(self, m, start=0, stop=None):
        """Return independent area-normalized source vectors for a source slice."""
        self._validate_model(m)
        stop = len(self.acquisition.sources_m) if stop is None else stop
        positions = self.acquisition.sources_m[start:stop]
        sampler = SamplingOperator(self.operator.grid, positions,
                                    boundary=self.operator.boundary)
        amplitudes = torch.tensor(self.acquisition.amplitudes[start:stop],
                                   dtype=self.observed.dtype, device=m.device)
        return sampler.source_rhs(torch.diag(amplitudes))

    def _iter_predictions(self, m, source_batch_size):
        self._validate_model(m)
        batch = self._batch_size(source_batch_size)
        ns = len(self.acquisition.sources_m)
        for fi, frequency in enumerate(self.frequencies_hz):
            for start in range(0, ns, batch):
                stop = min(start+batch, ns)
                rhs = self.source_rhs(m, start, stop)
                wave = implicit_solve(m, rhs, 2*math.pi*frequency,
                                      self.operator, self.solver)
                yield fi, start, stop, self.receivers.sample(wave)

    def predict(self, m, source_batch_size=None):
        """Return complex data [frequency,source,receiver], with a live graph."""
        by_frequency = [[] for _ in self.frequencies_hz]
        for fi, _, _, data in self._iter_predictions(m, source_batch_size):
            by_frequency[fi].append(data)
        return torch.stack([torch.cat(chunks, dim=0) for chunks in by_frequency])

    def iter_data_losses(self, m, source_batch_size=None):
        """Yield data-only scalar contributions in frequency/source-batch order."""
        for fi, start, stop, data in self._iter_predictions(m, source_batch_size):
            residual = self.weights[fi, start:stop]*(data-self.observed[fi, start:stop])
            # abs() is nonsmooth at zero; its autograd convention would lose
            # the exact-fit Hessian. This polynomial remains smooth there.
            yield 0.5*(residual.real.square()+residual.imag.square()).sum()

    def data_loss(self, m, source_batch_size=None):
        return sum(self.iter_data_losses(m, source_batch_size))

    def loss(self, m, source_batch_size=None):
        data = self.data_loss(m, source_batch_size)
        return data if self.regularizer is None else data+self.regularizer.value(m)
