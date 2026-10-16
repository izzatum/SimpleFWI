"""Atomic publication and full native optimizer restart equivalence."""
import random
import pickle

import numpy as np
import pytest
import torch

from simplefwi.fwi2d.checkpoint import (
    capture_checkpoint, load_checkpoint, restore_checkpoint, save_checkpoint,
)

CONFIG = {'parameterization': 'velocity', 'dataset_hash': 'abc',
          'model_hash': 'grid-v1', 'config_hash': 'cfg', 'solver_tolerances': {'rtol': 1e-9}}


def optimize(model, optimizer, scheduler, count):
    for _ in range(count):
        optimizer.zero_grad()
        loss = ((model.weight - 2) ** 2).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()


def make_run():
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(.4)
    optimizer = torch.optim.Adam(model.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.8)
    return model, optimizer, scheduler


def test_restart_equivalent_model_native_state_schedule_and_rng(tmp_path):
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    model, optimizer, scheduler = make_run()
    optimize(model, optimizer, scheduler, 3)
    state = capture_checkpoint(model, optimizer, scheduler, accepted_step=3, config=CONFIG,
                               metadata={'seed': 9, 'git_revision': 'fixture'})
    path = tmp_path / 'accepted.pt'
    save_checkpoint(path, state)
    draws = (random.random(), np.random.random(), torch.rand(3))
    optimize(model, optimizer, scheduler, 4)
    resumed, resumed_optimizer, resumed_scheduler = make_run()
    loaded = load_checkpoint(path, expected_config=CONFIG)
    restore_checkpoint(loaded, resumed, resumed_optimizer, resumed_scheduler)
    assert (random.random(), np.random.random()) == draws[:2]
    torch.testing.assert_close(torch.rand(3), draws[2], rtol=0, atol=0)
    optimize(resumed, resumed_optimizer, resumed_scheduler, 4)
    torch.testing.assert_close(resumed.weight, model.weight, rtol=0, atol=0)
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    assert resumed_optimizer.param_groups[0]['lr'] == optimizer.param_groups[0]['lr']
    for key, value in optimizer.state[model.weight].items():
        torch.testing.assert_close(resumed_optimizer.state[resumed.weight][key], value, rtol=0, atol=0)
    assert loaded['accepted_step'] == 3
    assert loaded['metadata']['seed'] == 9
    assert all(not v.requires_grad for v in loaded['model'].values())


@pytest.mark.parametrize('key,value', [('parameterization', 'log_velocity'),
                                     ('dataset_hash', 'other'), ('model_hash', 'other'),
                                     ('config_hash', 'other')])
def test_mismatched_restart_config_rejected(tmp_path, key, value):
    model, opt, sched = make_run()
    path = tmp_path / 'state.pt'
    save_checkpoint(path, capture_checkpoint(model, opt, sched, accepted_step=0, config=CONFIG))
    with pytest.raises(ValueError, match=key):
        load_checkpoint(path, expected_config={**CONFIG, key: value})


def test_optimizer_class_and_model_shapes_checked_before_restoring(tmp_path):
    model, opt, sched = make_run()
    state = capture_checkpoint(model, opt, sched, accepted_step=0, config=CONFIG)
    with pytest.raises(ValueError, match='optimizer'):
        restore_checkpoint(state, model, torch.optim.SGD(model.parameters(), lr=.1), sched)
    bad = torch.nn.Linear(3, 1, bias=False, dtype=torch.float64)
    with pytest.raises(ValueError, match='shape'):
        restore_checkpoint(state, bad, torch.optim.Adam(bad.parameters()), sched)


def test_interrupted_write_preserves_last_complete_checkpoint(tmp_path, monkeypatch):
    model, opt, sched = make_run()
    path = tmp_path / 'state.pt'
    state = capture_checkpoint(model, opt, sched, accepted_step=2, config=CONFIG)
    save_checkpoint(path, state)
    old = path.read_bytes()
    def interrupted(value, handle):
        handle.write(b'incomplete')
        raise OSError('injected interruption')
    monkeypatch.setattr(torch, 'save', interrupted)
    with pytest.raises(OSError, match='interruption'):
        save_checkpoint(path, {**state, 'accepted_step': 3})
    assert path.read_bytes() == old
    assert load_checkpoint(path, CONFIG)['accepted_step'] == 2
    assert list(tmp_path.iterdir()) == [path]


def test_schema_and_required_identity_validation(tmp_path):
    model, opt, sched = make_run()
    with pytest.raises(ValueError, match='dataset_hash'):
        capture_checkpoint(model, opt, sched, accepted_step=1,
                           config={'parameterization': 'velocity'})
    state = capture_checkpoint(model, opt, sched, accepted_step=1, config=CONFIG)
    state['schema_version'] = 999
    with pytest.raises(ValueError, match='schema'):
        save_checkpoint(tmp_path / 'bad.pt', state)


def test_captured_state_is_independent_of_later_updates(tmp_path):
    model, opt, sched = make_run()
    optimize(model, opt, sched, 1)
    state = capture_checkpoint(model, opt, sched, accepted_step=1, config=CONFIG)
    before = state['model']['weight'].clone()
    moment = state['optimizer']['state'][0]['exp_avg'].clone()
    optimize(model, opt, sched, 1)
    torch.testing.assert_close(state['model']['weight'], before, rtol=0, atol=0)
    torch.testing.assert_close(state['optimizer']['state'][0]['exp_avg'], moment, rtol=0, atol=0)


def test_atomic_replace_failure_preserves_last_checkpoint(tmp_path, monkeypatch):
    import os
    model, opt, sched = make_run()
    path = tmp_path / 'state.pt'
    state = capture_checkpoint(model, opt, sched, accepted_step=2, config=CONFIG)
    save_checkpoint(path, state)
    old = path.read_bytes()
    def interrupted(source, target):
        raise OSError('injected rename interruption')
    monkeypatch.setattr(os, 'replace', interrupted)
    with pytest.raises(OSError, match='rename interruption'):
        save_checkpoint(path, {**state, 'accepted_step': 3})
    assert path.read_bytes() == old
    assert load_checkpoint(path, CONFIG)['accepted_step'] == 2
    assert list(tmp_path.iterdir()) == [path]


def test_model_precision_and_scheduler_class_checked(tmp_path):
    model, opt, sched = make_run()
    state = capture_checkpoint(model, opt, sched, accepted_step=1, config=CONFIG)
    wrong = torch.nn.Linear(2, 1, bias=False, dtype=torch.float32)
    with pytest.raises(ValueError, match='dtype'):
        restore_checkpoint(state, wrong, torch.optim.Adam(wrong.parameters()), sched)
    with pytest.raises(ValueError, match='scheduler'):
        restore_checkpoint(state, model, opt)
    path = tmp_path / 'state.pt'
    save_checkpoint(path, state)
    with pytest.raises(ValueError, match='optimizer_class'):
        load_checkpoint(path, {**CONFIG, 'optimizer_class': 'torch.optim.SGD'})


def test_numpy_provenance_is_restricted_loader_compatible(tmp_path):
    model, opt, sched = make_run()
    path = tmp_path / 'state.pt'
    state = capture_checkpoint(model, opt, sched, accepted_step=1, config=CONFIG)
    save_checkpoint(path, state)
    state['accepted_step'] = 2
    state['metadata']['seed'] = np.int64(7)
    state['metadata']['coordinates'] = np.array([1., 2.])
    state['config']['nested'] = {'count': np.int32(3)}
    save_checkpoint(path, state)
    loaded = load_checkpoint(path, CONFIG)
    assert loaded['metadata']['seed'] == 7
    assert type(loaded['metadata']['seed']) is int
    assert loaded['metadata']['coordinates'] == [1., 2.]
    assert loaded['config']['nested'] == {'count': 3}


@pytest.mark.parametrize('location', ['metadata', 'config', 'optimizer'])
def test_unsupported_checkpoint_leaf_preserves_last_readable_file(tmp_path, location):
    model, opt, sched = make_run()
    path = tmp_path / 'state.pt'
    state = capture_checkpoint(model, opt, sched, accepted_step=1, config=CONFIG)
    save_checkpoint(path, state)
    before = path.read_bytes()
    state['accepted_step'] = 2
    state[location]['unsupported'] = object()
    with pytest.raises(TypeError, match='unsupported checkpoint value'):
        save_checkpoint(path, state)
    assert path.read_bytes() == before
    assert load_checkpoint(path, CONFIG)['accepted_step'] == 1
    assert list(tmp_path.iterdir()) == [path]


def test_numpy_config_expected_values_normalize_consistently(tmp_path):
    model, opt, sched = make_run()
    config = {**CONFIG, 'frequencies': np.array([2., 4.])}
    state = capture_checkpoint(model, opt, sched, accepted_step=1, config=config,
                               metadata={'seed': np.int64(7)})
    assert type(state['metadata']['seed']) is int
    path = tmp_path / 'state.pt'
    save_checkpoint(path, state)
    loaded = load_checkpoint(path, config)
    assert loaded['config']['frequencies'] == [2., 4.]


def test_unreadable_staged_serialization_preserves_last_good(tmp_path, monkeypatch):
    model, opt, sched = make_run()
    path = tmp_path / 'state.pt'
    state = capture_checkpoint(model, opt, sched, accepted_step=1, config=CONFIG)
    save_checkpoint(path, state)
    before = path.read_bytes()
    monkeypatch.setattr(torch, 'save', lambda value, handle: handle.write(b'not a torch file'))
    with pytest.raises(pickle.UnpicklingError):
        save_checkpoint(path, {**state, 'accepted_step': 2})
    assert path.read_bytes() == before
    assert load_checkpoint(path, CONFIG)['accepted_step'] == 1
    assert list(tmp_path.iterdir()) == [path]


def _assert_restart_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_restart_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for value, baseline in zip(actual, expected):
            _assert_restart_tree_equal(value, baseline)
    else:
        assert actual == expected


@pytest.mark.parametrize('representation', ['dense', 'sparse_embedding'])
def test_velocity_parameterization_actual_mapping_roundtrip(tmp_path, representation):
    from simplefwi.fwi2d.parameters import VelocityParameters
    model = VelocityParameters([[1600., 1700.]], (1400., 2200.), representation=representation)
    optimizer_type = torch.optim.Adam if representation == 'dense' else torch.optim.SparseAdam
    optimizer = optimizer_type(model.parameters(), lr=.05)
    ((model.velocity() - 1800.) / 1000.).square().sum().backward()
    optimizer.step()
    state = capture_checkpoint(model, optimizer, accepted_step=1, config=CONFIG)
    assert state['model_parameterization']['bounds_m_s'] == [1400., 2200.]
    assert state['model_parameterization']['representation'] == representation
    path = tmp_path / 'velocity.pt'
    save_checkpoint(path, state)
    resumed = VelocityParameters([[1550., 1650.]], (1400., 2200.), representation=representation)
    resumed_optimizer = optimizer_type(resumed.parameters(), lr=.01)
    restore_checkpoint(load_checkpoint(path, CONFIG), resumed, resumed_optimizer)
    torch.testing.assert_close(resumed.velocity(), model.velocity(), rtol=0, atol=0)
    torch.testing.assert_close(resumed.raw_parameter, model.raw_parameter, rtol=0, atol=0)
    _assert_restart_tree_equal(resumed_optimizer.state_dict(), optimizer.state_dict())


def test_velocity_bounds_mismatch_rejected_before_model_optimizer_scheduler_or_rng_mutation():
    from simplefwi.fwi2d.parameters import VelocityParameters
    original = VelocityParameters([[1600., 1700.]], (1400., 2200.))
    original_optimizer = torch.optim.Adam(original.parameters(), lr=.1)
    original_scheduler = torch.optim.lr_scheduler.StepLR(original_optimizer, step_size=1)
    state = capture_checkpoint(original, original_optimizer, original_scheduler,
                               accepted_step=1, config=CONFIG)
    target = VelocityParameters([[1900., 2000.]], (1000., 3000.))
    optimizer = torch.optim.Adam(target.parameters(), lr=.03)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    (target.raw_parameter.square().sum()).backward()
    optimizer.step()
    scheduler.step()
    baseline = capture_checkpoint(target, optimizer, scheduler, accepted_step=4, config=CONFIG)
    baseline_velocity = target.velocity().detach().clone()
    with pytest.raises(ValueError, match='model parameterization'):
        restore_checkpoint(state, target, optimizer, scheduler)
    after = capture_checkpoint(target, optimizer, scheduler, accepted_step=4, config=CONFIG)
    _assert_restart_tree_equal(after, baseline)
    torch.testing.assert_close(target.velocity(), baseline_velocity, rtol=0, atol=0)


def test_velocity_parameterization_metadata_required_and_representation_checked():
    from simplefwi.fwi2d.parameters import VelocityParameters
    source = VelocityParameters([[1600., 1700.]], (1400., 2200.))
    optimizer = torch.optim.Adam(source.parameters())
    state = capture_checkpoint(source, optimizer, accepted_step=1, config=CONFIG)
    sparse = VelocityParameters([[1600., 1700.]], (1400., 2200.), representation='sparse_embedding')
    with pytest.raises(ValueError, match='model parameterization'):
        restore_checkpoint(state, sparse, torch.optim.SparseAdam(sparse.parameters()))
    legacy = {key: value for key, value in state.items() if key != 'model_parameterization'}
    with pytest.raises(ValueError, match='model parameterization'):
        restore_checkpoint(legacy, source, optimizer)
