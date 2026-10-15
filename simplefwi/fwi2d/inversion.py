"""Deterministic native optimizer closures and accepted outer-step lifecycle."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import numbers
from typing import Any
import torch

from .optimizers import validate_optimizer


@dataclass(frozen=True)
class StepResult:
    accepted_step: int
    loss: float
    closure_evaluations: int
    diagnostics: dict[str, Any] = field(default_factory=dict)
    accepted: bool = True

    @property
    def iteration(self):
        return self.accepted_step


@dataclass(frozen=True)
class InversionResult:
    history: tuple[StepResult, ...]
    velocity: torch.Tensor
    slowness_squared: torch.Tensor


def _finite_scalar(loss):
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or loss.is_complex():
        raise ValueError('objective must return a real scalar tensor')
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('nonfinite objective; optimizer step is not accepted')


def _finite_state(value):
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('nonfinite native optimizer state')
    elif isinstance(value, dict):
        for item in value.values():
            _finite_state(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_state(item)


class InversionRunner:
    """Batch physical gradients, then pull back once into native parameters.

    Each source batch graph is freed immediately. A detached physical leaf is
    shared across batches; only its dense physical gradient persists. The full
    parameter transform is built once for the final pullback, including sparse
    embedding gradients. Closures return detached scalars and clear gradients.
    Source/frequency/weight/stage configuration must remain fixed during step.
    """
    def __init__(self, objective, model, optimizer, *, source_batch_size=1,
                 scheduler=None, recorder=None, accepted_callbacks=()):
        if (isinstance(source_batch_size, bool) or
            not isinstance(source_batch_size, numbers.Integral) or source_batch_size < 1):
            raise ValueError('source_batch_size must be a positive integer; default is bounded at one')
        self.objective, self.model = objective, model
        self.optimizer = optimizer if isinstance(optimizer, torch.optim.Optimizer) else optimizer(model.parameters())
        validate_optimizer(self.optimizer, model)
        if scheduler is not None and getattr(scheduler, 'optimizer', self.optimizer) is not self.optimizer:
            raise ValueError('scheduler must belong to the runner optimizer')
        self.scheduler, self.recorder = scheduler, recorder
        self.accepted_callbacks = tuple(accepted_callbacks)
        if any(not callable(callback) for callback in self.accepted_callbacks):
            raise TypeError('accepted_callbacks must contain callables')
        self.source_batch_size = int(source_batch_size)
        self.accepted_steps = 0
        self.closure_evaluations = 0
        self.last_physical_gradient = None

    @torch.enable_grad()
    def closure(self):
        self.optimizer.zero_grad(set_to_none=True)
        m = self.model.slowness_squared().detach().requires_grad_()
        total = torch.zeros((), dtype=m.dtype, device=m.device)
        batches = 0
        for loss in self.objective.iter_data_losses(m, self.source_batch_size):
            _finite_scalar(loss)
            total.add_(loss.detach())
            if loss.requires_grad:
                loss.backward()
            batches += 1
        if not batches:
            raise ValueError('objective must contain at least one source/frequency batch')
        regularizer = getattr(self.objective, 'regularizer', None)
        if regularizer is not None:
            penalty = regularizer.value(m)
            _finite_scalar(penalty)
            total.add_(penalty.detach())
            if penalty.requires_grad:
                penalty.backward()
        gradient = torch.zeros_like(m) if m.grad is None else m.grad
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError('nonfinite physical gradient')
        self.model.slowness_squared().backward(gradient)
        for p in self.model.parameters():
            values = p.grad.coalesce().values() if p.grad.is_sparse else p.grad
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError('nonfinite parameter gradient')
        self.last_physical_gradient = gradient.detach().clone()
        self.closure_evaluations += 1
        return total

    def _accepted_loss(self):
        with torch.no_grad():
            # Stream the same objective at the actual parameters left by step.
            # Do not trust native step's return value (LBFGS returns orig_loss).
            m = self.model.slowness_squared()
            total = torch.zeros((), dtype=m.dtype, device=m.device)
            for value in self.objective.iter_data_losses(m, self.source_batch_size):
                _finite_scalar(value)
                total.add_(value)
            regularizer = getattr(self.objective, 'regularizer', None)
            if regularizer is not None:
                total.add_(regularizer.value(m))
            _finite_scalar(total)
            return float(total)

    def step(self):
        # Rollback numerical failures, including an invalid final native trial.
        parameters = [p.detach().clone() for p in self.model.parameters()]
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        evaluations = self.closure_evaluations
        try:
            self.optimizer.step(self.closure)
            for p in self.model.parameters():
                if not bool(torch.isfinite(p).all()):
                    raise FloatingPointError('nonfinite optimizer parameter')
            _finite_state(self.optimizer.state)
            loss = self._accepted_loss()
        except BaseException:
            with torch.no_grad():
                for p, previous in zip(self.model.parameters(), parameters):
                    p.copy_(previous)
            self.optimizer.load_state_dict(optimizer_state)
            self.optimizer.zero_grad(set_to_none=True)
            self.last_physical_gradient = None
            raise
        number = self.accepted_steps+1
        diagnostics = self.model.diagnostics()
        diagnostics['learning_rates'] = [float(g['lr']) for g in self.optimizer.param_groups]
        diagnostics['weight_decay'] = [float(g.get('weight_decay', 0.)) for g in self.optimizer.param_groups]
        diagnostics['optimizer_class'] = type(self.optimizer).__name__
        result = StepResult(number, loss, self.closure_evaluations-evaluations, diagnostics)
        # The numerical update is now accepted. Commit its lifecycle before
        # external archival work, whose failure cannot undo the optimizer step.
        self.accepted_steps = number
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(loss)
            else:
                self.scheduler.step()
        # Recording payloads contain accepted model state only. Gradients from
        # the last closure may belong to another trial, so are never mislabeled.
        if self.recorder is not None:
            from .recording import ArrayPayload
            grid = getattr(getattr(self.objective, 'operator', None), 'grid', None)
            origin = getattr(grid, 'origin_m', None)
            payload = {'velocity': ArrayPayload(self.model.velocity().detach(), ('z', 'x'), 'm/s', origin=origin, region='physical'),
                       'slowness_squared': ArrayPayload(self.model.slowness_squared().detach(), ('z', 'x'), 's^2/m^2', origin=origin, region='physical')}
            self.recorder.record_accepted(result, payload)
        # These callbacks observe the actual accepted model and updated scheduler.
        # Archival failure propagates while preserving the accepted step counter.
        for callback in self.accepted_callbacks:
            callback(self, result)
        return result

    def run(self, steps):
        if isinstance(steps, bool) or not isinstance(steps, numbers.Integral) or steps < 0:
            raise ValueError('steps must be a nonnegative integer')
        history = tuple(self.step() for _ in range(steps))
        return InversionResult(history, self.model.velocity().detach().clone(),
                               self.model.slowness_squared().detach().clone())
