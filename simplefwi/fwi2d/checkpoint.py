"""Atomic local checkpoints for native PyTorch models, optimizers and RNGs.

Files contain only tensors and built-in containers and load with weights_only.
Atomic rename publishes one committed file; a separate mutable manifest is not
required. Identity hashes are supplied by the application, then checked on load.
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Mapping

import numpy as np
import torch

SCHEMA_VERSION = 1
_IDENTITIES = ('parameterization', 'dataset_hash', 'model_hash', 'config_hash')


def _class_name(obj):
    return f'{type(obj).__module__}.{type(obj).__qualname__}'


def _model_parameterization(model):
    """Capture physical mapping attributes absent from tensor state_dicts."""
    from .parameters import VelocityParameters
    if not isinstance(model, VelocityParameters):
        return None
    return {
        'class': _class_name(model),
        'transform': 'bounded_velocity_sigmoid',
        'bounds_m_s': [float(bound) for bound in model.bounds],
        'representation': model.representation,
    }


def _snapshot(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {key: _snapshot(value) for key, value in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_snapshot(value) for value in obj)
    if isinstance(obj, list):
        return [_snapshot(value) for value in obj]
    return copy.deepcopy(obj)



def _restricted_tree(value, location='state'):
    """Normalize provenance and reject leaves unsupported by weights_only.

    Rebuild containers as plain built-ins so custom pickle reducers cannot enter
    the staged checkpoint through caller config, metadata or native state.
    """
    if type(value) in (type(None), bool, int, float, complex, str, bytes):
        return value
    if isinstance(value, np.generic):
        native = value.item()
        if isinstance(native, np.generic):
            raise TypeError(f'unsupported checkpoint value at {location}: {type(value).__name__}')
        return _restricted_tree(native, location)
    if isinstance(value, np.ndarray):
        return _restricted_tree(value.tolist(), location)
    if isinstance(value, torch.Tensor):
        if type(value) not in (torch.Tensor, torch.nn.Parameter):
            raise TypeError(f'unsupported checkpoint value at {location}: tensor subclass {type(value).__name__}')
        return value.detach()
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized_key = _restricted_tree(key, f'{location}.<key>')
            try:
                hash(normalized_key)
            except TypeError as exc:
                raise TypeError(f'unsupported checkpoint value at {location}: unhashable key') from exc
            result[normalized_key] = _restricted_tree(item, f'{location}.{normalized_key}')
        return result
    if isinstance(value, tuple):
        return tuple(_restricted_tree(item, f'{location}[{index}]')
                     for index, item in enumerate(value))
    if isinstance(value, list):
        return [_restricted_tree(item, f'{location}[{index}]')
                for index, item in enumerate(value)]
    raise TypeError(f'unsupported checkpoint value at {location}: {type(value).__name__}')


def _config(config):
    for key in _IDENTITIES:
        if key not in config or not config[key]:
            raise ValueError(f'restart config requires {key}')


def _validate(state):
    if state.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported checkpoint schema_version')
    for key in ('model', 'optimizer', 'scheduler', 'rng', 'config', 'metadata',
                'accepted_step', 'optimizer_class', 'scheduler_class', 'model_shapes',
                'model_dtypes', 'optimizer_shapes'):
        if key not in state:
            raise ValueError(f'checkpoint missing {key}')
    parameterization = state.get('model_parameterization')
    if parameterization is not None:
        if (not isinstance(parameterization, dict) or
                not isinstance(parameterization.get('class'), str) or
                parameterization.get('transform') != 'bounded_velocity_sigmoid' or
                parameterization.get('representation') not in ('dense', 'sparse_embedding')):
            raise ValueError('invalid model parameterization metadata')
        bounds = parameterization.get('bounds_m_s')
        if (not isinstance(bounds, list) or len(bounds) != 2 or
                any(type(value) not in (int, float) or not math.isfinite(value) for value in bounds) or
                not 0 < bounds[0] < bounds[1]):
            raise ValueError('invalid model parameterization velocity bounds')
    _config(state['config'])
    if not isinstance(state['accepted_step'], int) or state['accepted_step'] < 0:
        raise ValueError('accepted_step must be a nonnegative integer')
    for key in ('python', 'numpy', 'torch', 'cuda'):
        if key not in state['rng']:
            raise ValueError(f'checkpoint missing RNG state {key}')


def capture_checkpoint(model, optimizer, scheduler=None, *, accepted_step, config, metadata=None):
    """Detach and clone a complete accepted state, including native optimizer state.

    Caller invokes this after an accepted outer step, never from a trial closure.
    CUDA RNG states are captured only for an already initialized CUDA runtime.
    """
    _config(config)
    numpy_rng = np.random.get_state()
    model_state = model.state_dict()
    state = {
        'schema_version': SCHEMA_VERSION,
        'accepted_step': accepted_step,
        'model': _snapshot(model_state),
        'model_parameterization': _model_parameterization(model),
        'model_shapes': {name: list(value.shape) for name, value in model_state.items()},
        'model_dtypes': {name: str(value.dtype) for name, value in model_state.items()},
        'optimizer': _snapshot(optimizer.state_dict()),
        'optimizer_class': _class_name(optimizer),
        'optimizer_shapes': [[list(p.shape) for p in group['params']] for group in optimizer.param_groups],
        'scheduler': None if scheduler is None else _snapshot(scheduler.state_dict()),
        'scheduler_class': None if scheduler is None else _class_name(scheduler),
        'rng': {
            'python': random.getstate(),
            'numpy': (numpy_rng[0], numpy_rng[1].tolist(), numpy_rng[2], numpy_rng[3], numpy_rng[4]),
            'torch': torch.get_rng_state().clone(),
            'cuda': [value.cpu().clone() for value in torch.cuda.get_rng_state_all()]
                    if torch.cuda.is_initialized() else [],
        },
        'config': copy.deepcopy(dict(config)),
        'metadata': {'torch_version': str(torch.__version__),
                     'numpy_version': str(np.__version__), **dict(metadata or {})},
    }
    state = _restricted_tree(state)
    _validate(state)
    return state


def save_checkpoint(path, state):
    """Publish a complete checkpoint by same-directory fsync + atomic os.replace.

    Failure before replace preserves the previous checkpoint and removes the
    temporary file. Local filesystem atomic rename semantics are required.
    """
    state = _restricted_tree(state)
    _validate(state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        staged = torch.load(temporary, map_location='cpu', weights_only=True)
        _validate(staged)
        del staged
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _expected(actual, expected, prefix='config'):
    for key, value in expected.items():
        label = f'{prefix}.{key}'
        if key not in actual:
            raise ValueError(f'checkpoint mismatch: missing {label}')
        if isinstance(value, Mapping):
            if not isinstance(actual[key], Mapping):
                raise ValueError(f'checkpoint mismatch: {label}')
            _expected(actual[key], value, label)
        elif actual[key] != value:
            raise ValueError(f'checkpoint mismatch: {label}')


def load_checkpoint(path, expected_config):
    """Load on CPU and validate schema and caller-supplied dataset/config identity.

    ``expected_config`` must include all four identity fields. It can additionally
    include any config values, and optimizer_class/scheduler_class/model_shapes.
    """
    expected_config = _restricted_tree(dict(expected_config), 'expected_config')
    _config(expected_config)
    state = torch.load(path, map_location='cpu', weights_only=True)
    _validate(state)
    config_expected = dict(expected_config)
    for key in ('optimizer_class', 'scheduler_class', 'model_shapes'):
        if key in config_expected:
            _expected(state, {key: config_expected.pop(key)}, prefix='checkpoint')
    _expected(state['config'], config_expected)
    return state


def restore_checkpoint(state, model, optimizer, scheduler=None, *, restore_rng=True):
    """Validate model precision/shapes and native classes before applying state.

    Returns the accepted outer-step counter. Parameters remain on the receiving
    model's device. Native optimizer.load_state_dict performs its supported state
    relocation; no optimizer-state format is reimplemented here.
    """
    _validate(state)
    if state.get('model_parameterization') != _model_parameterization(model):
        raise ValueError('model parameterization mismatch: physical velocity bounds, representation or class')
    current = model.state_dict()
    shapes = {name: list(value.shape) for name, value in current.items()}
    if shapes != state['model_shapes']:
        raise ValueError('model shape mismatch')
    dtypes = {name: str(value.dtype) for name, value in current.items()}
    if dtypes != state['model_dtypes']:
        raise ValueError('model dtype mismatch')
    if _class_name(optimizer) != state['optimizer_class']:
        raise ValueError('optimizer class mismatch')
    optimizer_shapes = [[list(p.shape) for p in group['params']] for group in optimizer.param_groups]
    if optimizer_shapes != state['optimizer_shapes']:
        raise ValueError('optimizer parameter group shape mismatch')
    scheduler_name = None if scheduler is None else _class_name(scheduler)
    if scheduler_name != state['scheduler_class']:
        raise ValueError('scheduler class mismatch')
    cuda_states = state['rng']['cuda']
    if restore_rng and cuda_states and (not torch.cuda.is_available() or
                                       len(cuda_states) != torch.cuda.device_count()):
        raise ValueError('CUDA RNG topology mismatch; restore_rng=False must be explicit to omit RNG restoration')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    if restore_rng:
        random.setstate(state['rng']['python'])
        nrng = state['rng']['numpy']
        np.random.set_state((nrng[0], np.asarray(nrng[1], dtype=np.uint32), nrng[2], nrng[3], nrng[4]))
        torch.set_rng_state(state['rng']['torch'])
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
    return state['accepted_step']
