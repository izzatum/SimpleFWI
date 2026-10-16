"""Accepted-state archival and restart adapters for explicit native runners."""
from __future__ import annotations

import numbers
from pathlib import Path
import torch

from .checkpoint import capture_checkpoint, save_checkpoint
from .recording import ArrayPayload


class AcceptedStateRecorder:
    """Callback recomputing optional physical gradients at accepted parameters.

    Checkpoints are independent of HDF5 snapshot cadence, default to every
    accepted outer step, and include the scheduler after its accepted update.
    Supply source/dataset/model/config hashes in ``config``. Recording errors
    propagate: an optimizer step may already be accepted when archival fails.
    The caller closes the Recorder to observe asynchronous writer failures.
    """
    def __init__(self, recorder, *, checkpoint_path=None, config=None, metadata=None,
                 checkpoint_cadence=1):
        if (isinstance(checkpoint_cadence, bool) or not isinstance(checkpoint_cadence, numbers.Integral)
                or checkpoint_cadence < 1):
            raise ValueError('checkpoint_cadence must be a positive integer')
        self.recorder = recorder
        self.checkpoint_path = (Path(str(recorder.policy.path)+'.checkpoint.pt')
                                if checkpoint_path is None else Path(checkpoint_path))
        if config is None:
            raise ValueError('checkpoint config with dataset/model/config identity hashes is required')
        self.config, self.metadata = dict(config), dict(metadata or {})
        for key in ('parameterization','dataset_hash','model_hash','config_hash'):
            if not self.config.get(key):
                raise ValueError(f'checkpoint config requires {key}')
        self.checkpoint_cadence = int(checkpoint_cadence)

    def __call__(self, runner, result):
        if not result.accepted or runner.accepted_steps != result.accepted_step:
            raise ValueError('callback requires the committed accepted runner state')
        model, objective = runner.model, runner.objective
        grid = objective.operator.grid
        coordinates = {'spacing_m': grid.spacing_m}
        payload = {
            'velocity': ArrayPayload(model.velocity().detach(), ('z','x'), 'm/s',
                                    origin=grid.origin_m, region='physical', coordinates=coordinates),
            'slowness_squared': ArrayPayload(model.slowness_squared().detach(), ('z','x'), 's^2/m^2',
                                    origin=grid.origin_m, region='physical', coordinates=coordinates),
        }
        if ('physical_gradient' in self.recorder.policy.fields and
                result.accepted_step % self.recorder.policy.cadence == 0):
            with torch.enable_grad():
                m = model.slowness_squared().detach().requires_grad_()
                for loss in objective.iter_data_losses(m, runner.source_batch_size):
                    loss.backward()
                if objective.regularizer is not None:
                    objective.regularizer.value(m).backward()
                gradient = torch.zeros_like(m) if m.grad is None else m.grad
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError('nonfinite accepted physical gradient')
            payload['physical_gradient'] = ArrayPayload(gradient.detach(), ('z','x'),
                'objective_unit/(s^2/m^2)', origin=grid.origin_m, region='physical', coordinates=coordinates)
        self.recorder.record_accepted(result, payload)
        if result.accepted_step % self.checkpoint_cadence == 0:
            state = capture_checkpoint(model, runner.optimizer, runner.scheduler,
                accepted_step=result.accepted_step, config=self.config, metadata=self.metadata)
            save_checkpoint(self.checkpoint_path, state)
