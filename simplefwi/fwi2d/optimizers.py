"""Executable recipes for every stable native optimizer in torch 2.12.0."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Any
import torch


@dataclass(frozen=True)
class OptimizerRecipe:
    optimizer_class: type[torch.optim.Optimizer]
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    representation: str = 'dense'
    constraints: str = 'Real two-dimensional dense parameter; native default flags.'

    def build(self, model, **kwargs):
        if model.representation != self.representation:
            raise ValueError(f'{self.optimizer_class.__name__} recipe requires {self.representation}')
        optimizer = self.optimizer_class(model.parameters(), **{**self.kwargs, **kwargs})
        validate_optimizer(optimizer, model)
        return optimizer


# Explicit inventory: a newly exported native class fails inventory CI.
OPTIMIZER_RECIPES = {
    'Adafactor': OptimizerRecipe(torch.optim.Adafactor, {'lr': 0.01}),
    'Adadelta': OptimizerRecipe(torch.optim.Adadelta, {'lr': 1.0}),
    'Adagrad': OptimizerRecipe(torch.optim.Adagrad, {'lr': 0.01}),
    'Adam': OptimizerRecipe(torch.optim.Adam, {'lr': 0.01}),
    'Adamax': OptimizerRecipe(torch.optim.Adamax, {'lr': 0.01}),
    'AdamW': OptimizerRecipe(torch.optim.AdamW, {'lr': 0.01, 'weight_decay': 0.0}),
    'ASGD': OptimizerRecipe(torch.optim.ASGD, {'lr': 0.01, 'lambd': 0.0}),
    'LBFGS': OptimizerRecipe(torch.optim.LBFGS,
                              {'lr': 1.0, 'max_iter': 5, 'history_size': 5,
                               'line_search_fn': 'strong_wolfe'}),
    'Muon': OptimizerRecipe(torch.optim.Muon, {'lr': 0.01, 'weight_decay': 0.0},
                             constraints='Dense 2D parameter, required by native Muon.'),
    'NAdam': OptimizerRecipe(torch.optim.NAdam, {'lr': 0.01}),
    'RAdam': OptimizerRecipe(torch.optim.RAdam, {'lr': 0.01}),
    'RMSprop': OptimizerRecipe(torch.optim.RMSprop, {'lr': 0.01}),
    'Rprop': OptimizerRecipe(torch.optim.Rprop, {'lr': 0.01}),
    'SGD': OptimizerRecipe(torch.optim.SGD, {'lr': 0.01}),
    'SparseAdam': OptimizerRecipe(torch.optim.SparseAdam, {'lr': 0.01},
                                  representation='sparse_embedding',
                                  constraints='Sparse-gradient active-cell embedding; no weight_decay.'),
}


def make_optimizer(name, model, **kwargs):
    """Forward native keyword arguments through the documented valid recipe."""
    if name not in OPTIMIZER_RECIPES:
        raise ValueError(f'unknown optimizer {name!r}; choose {sorted(OPTIMIZER_RECIPES)}')
    return OPTIMIZER_RECIPES[name].build(model, **kwargs)


def validate_optimizer(optimizer, model):
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError('optimizer factory must return a native torch.optim.Optimizer')
    supplied = [p for group in optimizer.param_groups for p in group['params']]
    expected = list(model.parameters())
    if len(supplied) != len(expected) or {id(p) for p in supplied} != {id(p) for p in expected}:
        raise ValueError('optimizer must own exactly the trainable model parameters')
    if isinstance(optimizer, torch.optim.SparseAdam):
        if model.representation != 'sparse_embedding':
            raise ValueError('SparseAdam requires representation="sparse_embedding"')
    elif model.representation == 'sparse_embedding':
        raise ValueError('sparse_embedding integration currently supports SparseAdam only')
    if isinstance(optimizer, torch.optim.Muon) and any(p.ndim != 2 for p in supplied):
        raise ValueError('Muon requires two-dimensional parameters')
    for group in optimizer.param_groups:
        if group.get('differentiable', False) or group.get('capturable', False):
            raise ValueError('differentiable/capturable optimizer steps are unsupported by this closure runner')
