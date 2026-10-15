"""Public workflow contracts, including an installed-package invocation."""
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import pytest
import torch

from simplefwi.fwi2d import (
    AcousticFWI, Acquisition, BoundaryConfig, Curvature, Grid2D,
    InversionRunner, Recorder, RecordingPolicy, SolverConfig, VelocityParameters,
    AcceptedStateRecorder, load_checkpoint, restore_checkpoint,
)


def fixture():
    grid = Grid2D((6, 7), (10., 10.))
    acq = Acquisition(((10., 10.), (10., 40.)), ((20., 15.), (20., 45.)))
    api = AcousticFWI(grid, acq, BoundaryConfig(pml_cells=3), SolverConfig())
    velocity = torch.full(grid.shape, 1500., dtype=torch.float64)
    return api, velocity


def test_native_step_hvp_record_restart(tmp_path):
    api, true = fixture()
    observed = api.simulate(true, (2.,), source_batch_size=1)
    assert observed.shape == (1, 2, 2)
    objective = api.objective(observed, (2.,))
    model = VelocityParameters(true+20, (1400., 1800.))
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    path = tmp_path/'accepted.h5'
    identity = {'parameterization': 'dense', 'dataset_hash': 'fixture',
                'model_hash': 'initial', 'config_hash': 'fixed'}
    with Recorder(RecordingPolicy(path, fields=('velocity', 'physical_gradient'))) as recorder:
        adapter = AcceptedStateRecorder(recorder, checkpoint_path=tmp_path/'state.pt', config=identity)
        runner = InversionRunner(objective, model, optimizer, source_batch_size=1,
                                 accepted_callbacks=(adapter,))
        result = runner.step()
    assert result.accepted_step == 1
    with h5py.File(path) as handle:
        group = handle['steps/000000000001']
        assert group.attrs['complete']
        torch.testing.assert_close(torch.from_numpy(group['velocity'][:]), model.velocity().detach())
        m = model.slowness_squared().detach().requires_grad_()
        expected = torch.autograd.grad(objective.loss(m), m)[0]
        torch.testing.assert_close(torch.from_numpy(group['physical_gradient'][:]), expected)
    curvature = Curvature(objective, model.slowness_squared(), source_batch_size=1)
    hvp = curvature.exact_hvp(torch.ones(curvature.n_active, dtype=torch.float64))
    assert hvp.shape == (curvature.n_active,) and torch.isfinite(hvp).all()
    state = load_checkpoint(tmp_path/'state.pt', identity)
    other = VelocityParameters(true+20, (1400., 1800.))
    other_optimizer = torch.optim.Adam(other.parameters(), lr=.01)
    assert restore_checkpoint(state, other, other_optimizer) == 1
    torch.testing.assert_close(other.velocity(), model.velocity())
    resumed = InversionRunner(objective, other, other_optimizer, source_batch_size=1)
    resumed.accepted_steps = 1
    runner.accepted_callbacks = ()
    runner.step()
    resumed.step()
    torch.testing.assert_close(other.velocity(), model.velocity(), rtol=0, atol=0)


def test_shape_frequency_precision_and_device_contracts():
    api, velocity = fixture()
    with pytest.raises(ValueError, match='shape'):
        api.simulate(velocity[:-1], (2.,), source_batch_size=1)
    with pytest.raises(ValueError, match='positive'):
        api.simulate(velocity, (0.,), source_batch_size=1)
    with pytest.raises(ValueError, match='precision'):
        api.simulate(velocity.float(), (2.,), source_batch_size=1)
    with pytest.raises(ValueError, match='CPU'):
        AcousticFWI(api.grid, api.acquisition, api.boundary, SolverConfig(device='cuda'))
    with pytest.raises(ValueError, match='shape'):
        api.objective(torch.zeros((2,2), dtype=torch.complex128), (2.,))


def test_selected_wavefield_axes(tmp_path):
    api, velocity = fixture()
    path = tmp_path/'waves.h5'
    with Recorder(RecordingPolicy(path, fields=('physical_wavefield', 'pml_wavefield'),
                                  sources=(1,), frequencies=(2.,))) as recorder:
        api.simulate(velocity, (2., 3.), source_batch_size=1, recorder=recorder)
    with h5py.File(path) as handle:
        groups = list(handle['steps'].values())
        assert len(groups) == 1
        wave = groups[0]['physical_wavefield']
        assert wave.shape == (1, 1, *api.grid.shape)
        assert json.loads(wave.attrs['axes']) == ['frequency','source','z','x']
        assert json.loads(groups[0]['pml_wavefield'].attrs['axes'])[-2:] == ['z_padded','x_padded']


def test_cli_outside_checkout(tmp_path):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    process = subprocess.run([sys.executable, '-m', 'simplefwi.fwi2d', '--output',
                              str(tmp_path/'run')], cwd=tmp_path, env=env,
                             text=True, capture_output=True, check=True)
    manifest = json.loads((tmp_path/'run/manifest.json').read_text())
    assert manifest['accepted_steps'] == 1
    assert manifest['accuracy']['validation_status'] == 'screened_only'
    assert manifest['accuracy']['physical_accuracy_validated'] is False
    assert 'manifest.json' in process.stdout


def test_callback_failure_keeps_accepted_counter(tmp_path, monkeypatch):
    import simplefwi.fwi2d.integration as integration
    api, true = fixture()
    objective = api.objective(api.simulate(true,(2.,),source_batch_size=1),(2.,))
    model = VelocityParameters(true+20,(1400.,1800.))
    optimizer = torch.optim.Adam(model.parameters(),lr=.01)
    config = {'parameterization':'dense','dataset_hash':'a','model_hash':'b','config_hash':'c'}
    def failed_save(*args):
        raise OSError('checkpoint storage unavailable')
    monkeypatch.setattr(integration,'save_checkpoint',failed_save)
    with Recorder(RecordingPolicy(tmp_path/'failure.h5')) as recorder:
        adapter = AcceptedStateRecorder(recorder,config=config)
        runner = InversionRunner(objective,model,optimizer,accepted_callbacks=(adapter,))
        with pytest.raises(OSError,match='checkpoint storage unavailable'):
            runner.step()
        assert runner.accepted_steps == 1
    assert not (tmp_path/'failure.h5.checkpoint.pt').exists()


def test_nested_surface_sources_and_explicit_units(tmp_path):
    from simplefwi.fwi2d import camembert_config, nested_surface_acquisition, run_experiment
    grid = Grid2D((6,9),(25.,25.),(100.,-200.))
    boundary = BoundaryConfig(pml_cells=3)
    previous = set()
    for count in (2,3,5,9):
        geometry = nested_surface_acquisition(grid,boundary,source_count=count)
        assert previous.issubset(set(geometry.sources_m))
        previous = set(geometry.sources_m)
    assert len(previous) == grid.shape[1]
    with pytest.raises(ValueError,match='exceeds'):
        nested_surface_acquisition(grid,boundary,source_count=10)
    config = camembert_config()
    config['units']['velocity'] = 'km/s'
    with pytest.raises(ValueError,match='units'):
        run_experiment(config,output=tmp_path/'invalid')
    config = camembert_config()
    config['frequencies_hz'] = [30.]
    with pytest.raises(ValueError,match='dispersion'):
        run_experiment(config,output=tmp_path/'dispersion-rejected')


def test_example_runs_outside_checkout(tmp_path):
    checkout = Path(__file__).resolve().parents[2]
    example = checkout/'examples/fwi2d/marmousi.py'
    config = checkout/'examples/fwi2d/configs/camembert.json'
    env = dict(os.environ)
    env.pop('PYTHONPATH',None)
    subprocess.run([sys.executable,str(example),'--config',str(config),
                    '--output',str(tmp_path/'example-run')],cwd=tmp_path,env=env,
                   text=True,capture_output=True,check=True)
    assert (tmp_path/'example-run/checkpoint.pt').exists()


def test_public_fgmres_matches_reference():
    direct, velocity = fixture()
    iterative = AcousticFWI(direct.grid,direct.acquisition,direct.boundary,
                            SolverConfig(method='fgmres',maxiter=100,restart=20))
    expected = direct.simulate(velocity,(2.,),source_batch_size=1)
    actual = iterative.simulate(velocity,(2.,),source_batch_size=1)
    torch.testing.assert_close(actual,expected,rtol=2e-9,atol=1e-10)
    assert all(report.converged for report in iterative.last_solve_reports)


def test_restart_memory_preflight_rejects_before_solver(tmp_path, monkeypatch):
    import simplefwi.fwi2d.experiments as experiments
    from simplefwi.fwi2d import PlanningError, camembert_config
    config = camembert_config()
    config['solver'].update(method='fgmres',restart=200,maxiter=200,
                            memory_budget_bytes=4_000_000)
    # Isolate the basis-size regression from the independent recorder pool.
    config['recording'].update(chunk_bytes=4096,queue_capacity=2)
    class SolverConstructionReached(RuntimeError):
        pass
    def construction_reached(*args,**kwargs):
        raise SolverConstructionReached('memory preflight passed')
    monkeypatch.setattr(experiments,'AcousticFWI',construction_reached)
    config['solver']['restart']=30
    with pytest.raises(SolverConstructionReached,match='preflight passed'):
        experiments.run_experiment(config,output=tmp_path/'restart30')
    config['solver']['restart']=200
    with pytest.raises(PlanningError,match='memory budget'):
        experiments.run_experiment(config,output=tmp_path/'rejected')


def test_manifest_uses_configured_memory_inputs(tmp_path):
    from simplefwi.fwi2d import camembert_config, run_experiment
    config = camembert_config()
    config['solver'].update(method='fgmres',restart=200,maxiter=200,
                            memory_budget_bytes=64_000_000)
    config['optimizer']={'name':'LBFGS','lr':.01,'history_size':7,'max_iter':1}
    config['steps']=0
    config['recording'].update(chunk_bytes=4096,queue_capacity=3)
    config['preflight']['extra_workspace_bytes']=12345
    manifest=run_experiment(config,output=tmp_path/'configured')
    estimate=manifest['resources']['memory_preflight']
    assert estimate['krylov_restart']==200
    assert estimate['optimizer_history_size']==7
    assert estimate['recording_buffer_bytes']==4096*3
    assert estimate['caller_extra_workspace_bytes']==12345
    assert estimate['total_extra_workspace_bytes']==4096*3+12345
    assert manifest['resources']['memory_budget_is_process_limit'] is False
    assert manifest['resources']['memory_estimate_exclusions']
