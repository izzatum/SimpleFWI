"""Reproducible, bounded experiment configuration used by the public CLI."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import resource
import time

import torch
import torch.nn.functional as F

from .api import AcousticFWI
from .checkpoint import load_checkpoint, restore_checkpoint
from .datasets import load_marmousi
from .inversion import InversionRunner
from .integration import AcceptedStateRecorder
from .parameters import VelocityParameters
from .planning import StageRequest, plan_stage
from .recording import Recorder, RecordingPolicy
from .types import Acquisition, BoundaryConfig, Grid2D, SolverConfig


def camembert_config():
    """Small algebraic smoke example; independent physical validation is separate."""
    return {
        'schema_version': 1,
        'units': {'coordinates': 'm', 'velocity': 'm/s', 'frequency': 'Hz'},
        'model': {'kind': 'camembert', 'grid': {'shape': [17,17], 'spacing_m': [10.,10.],
                    'origin_m': [0.,0.]}, 'centre_m': [80.,80.], 'radius_m': 30.,
                    'background_m_s': 1500., 'inclusion_m_s': 1700.},
        'boundary': {'top': 'pml', 'pml_cells': 8, 'sigma_max': 80., 'alpha': 0.},
        'acquisition': {'sources_m': [[20.,30.],[20.,130.]],
                        'receivers_m': [[30.,30.],[30.,60.],[30.,100.],[30.,130.]]},
        'frequencies_hz': [2.], 'velocity_bounds_m_s': [1400.,2000.],
        'solver': {'method': 'direct', 'device': 'cpu', 'dtype': 'complex128',
                   'rtol': 1e-10, 'atol': 1e-12, 'memory_budget_bytes': 134217728},
        'source_batch_size': 1, 'optimizer': {'name': 'Adam', 'lr': .02}, 'steps': 1,
        'recording': {'fields': ['velocity','slowness_squared'], 'checkpoint_cadence': 1},
        'preflight': {'max_compact_dof': 10000, 'max_forward_rhs': 32,
                      'max_phase_error': .01, 'classification': 'algebraic_smoke'},
        'seed': 17,
    }


def nested_surface_acquisition(grid, boundary, *, source_count, source_depth_index=1,
                               receiver_depth_index=2, receiver_stride=1):
    """Nested dyadic source order, reaching one source per usable x column.

    Coordinates retain explicit grid origin. Two sources cover the aperture
    endpoints; subsequent sources repeatedly bisect the largest remaining gap.
    Source densification alone does not imply additional independent information.
    """
    count = _positive_count(source_count, 'source_count')
    stride = _positive_count(receiver_stride, 'receiver_stride')
    first, last = (0,grid.shape[1]-1) if boundary.pml_cells else (1,grid.shape[1]-2)
    columns = list(range(first,last+1))
    if count > len(columns):
        raise ValueError('source_count exceeds one per usable horizontal grid column')
    order = [first] if first == last else [first,last]
    intervals = [(first,last)]
    while len(order) < count:
        intervals.sort(key=lambda interval: (-(interval[1]-interval[0]),interval[0]))
        left,right = intervals.pop(0)
        midpoint = (left+right)//2
        if right-left <= 1:
            raise ValueError('no additional usable source column')
        order.append(midpoint)
        intervals.extend([(left,midpoint),(midpoint,right)])
    def positions(depth, indices):
        if isinstance(depth,bool) or not isinstance(depth,int) or not 0 <= depth < grid.shape[0]:
            raise ValueError('source/receiver depth index must lie inside the physical grid')
        return tuple((grid.origin_m[0]+depth*grid.spacing_m[0],
                      grid.origin_m[1]+column*grid.spacing_m[1]) for column in indices)
    return Acquisition(positions(source_depth_index,order[:count]),
                       positions(receiver_depth_index,columns[::stride]))


def _hash_tensor(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _positive_count(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f'{name} must be an integer >= {0 if allow_zero else 1}')
    return value


def _model(config, base_path):
    description = config['model']
    kind = description['kind']
    if kind == 'camembert':
        grid = Grid2D(**description['grid'])
        if math.prod(grid.shape) > 10000:
            raise ValueError('smoke preflight rejects more than 10000 physical cells')
        z = grid.origin_m[0]+torch.arange(grid.shape[0], dtype=torch.float64)*grid.spacing_m[0]
        x = grid.origin_m[1]+torch.arange(grid.shape[1], dtype=torch.float64)*grid.spacing_m[1]
        centre = description['centre_m']
        inside = ((z[:,None]-centre[0])**2+(x[None,:]-centre[1])**2 <= description['radius_m']**2)
        initial = torch.full(grid.shape, description['background_m_s'], dtype=torch.float64)
        true = torch.where(inside, description['inclusion_m_s'], initial)
        provenance = {'kind': kind, 'classification': 'same_grid_algebraic_data'}
    elif kind == 'marmousi':
        path = Path(description['path'])
        if not path.is_absolute():
            path = base_path/path
        grid, true = load_marmousi(path, velocity_unit=description['velocity_unit'])
        provenance = {'kind': kind, 'path': str(path.resolve()),
                      'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'input_velocity_unit': description['velocity_unit'],
                      'classification': 'same_grid_algebraic_data'}
        if 'crop_indices' in description:
            z0,z1,x0,x1 = description['crop_indices']
            if (any(isinstance(i,bool) or not isinstance(i,int) for i in (z0,z1,x0,x1))
                    or not (0 <= z0 < z1 <= grid.shape[0] and 0 <= x0 < x1 <= grid.shape[1])):
                raise ValueError('crop_indices must be valid [z_start,z_stop,x_start,x_stop]')
            true = true[z0:z1,x0:x1].clone()
            grid = Grid2D(tuple(true.shape), grid.spacing_m,
                          (grid.origin_m[0]+z0*grid.spacing_m[0],grid.origin_m[1]+x0*grid.spacing_m[1]))
            provenance['crop_indices'] = [z0,z1,x0,x1]
        if 'spacing_m' in description:
            spacing = float(description['spacing_m'])
            if spacing not in (25.,12.5,6.25):
                raise ValueError('Marmousi computational candidate spacing must be 25, 12.5 or 6.25 m')
            extent = [(n-1)*h for n,h in zip(grid.shape,grid.spacing_m)]
            shape = tuple(round(length/spacing)+1 for length in extent)
            if any(not math.isclose((n-1)*spacing,length,abs_tol=1e-9) for n,length in zip(shape,extent)):
                raise ValueError('requested spacing must preserve node extent exactly')
            if math.prod(shape) > 10000:
                raise ValueError('smoke preflight rejects more than 10000 resampled physical cells')
            true = F.interpolate(true[None,None], size=shape, mode='bilinear', align_corners=True)[0,0]
            grid = Grid2D(shape, (spacing,spacing), grid.origin_m)
            provenance['interpolation'] = 'bilinear; no added geological resolution'
        # A changed reference ensures the smoke gradient is nonzero even in water.
        perturbation = float(description.get('initial_scale', .98))
        initial = true*perturbation
    else:
        raise ValueError('model.kind must be camembert or marmousi')
    return grid, true, initial, provenance


def run_experiment(config=None, *, output, base_path='.', resume=None):
    """Execute one bounded algebraic workflow and publish an inspectable manifest.

    Only smoke experiments are authorized by these built-in preflights. A
    passing StagePlan is spatial/resource screening, never evidence of PML,
    heterogeneous physical accuracy, inversion recovery or marginal source
    information. Larger research cases need independent accuracy validation.
    Native models, optimizers and runners remain directly available to users.
    """
    config = camembert_config() if config is None else json.loads(json.dumps(config))
    if config.get('schema_version') != 1:
        raise ValueError('unsupported experiment schema_version')
    if config.get('units') != {'coordinates':'m','velocity':'m/s','frequency':'Hz'}:
        raise ValueError('explicit experiment units must be coordinates=m, velocity=m/s, frequency=Hz')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config.get('seed',17))
    started = time.perf_counter()
    grid, true, initial, provenance = _model(config, Path(base_path))
    bounds = tuple(config['velocity_bounds_m_s'])
    if not bool(((true >= bounds[0]) & (true <= bounds[1])).all()):
        raise ValueError('true SI velocity must lie within declared velocity_bounds_m_s')
    boundary = BoundaryConfig(**config['boundary'])
    geometry = dict(config['acquisition'])
    if geometry.pop('kind',None) == 'nested_surface':
        acquisition = nested_surface_acquisition(grid,boundary,**geometry)
    else:
        acquisition = Acquisition(**geometry)
    serialized_solver = dict(config['solver'])
    dtype = serialized_solver.pop('dtype', 'complex128')
    if dtype not in ('complex128','complex64'):
        raise ValueError('solver.dtype must explicitly be complex128 or complex64')
    solver_config = SolverConfig(**serialized_solver, dtype=getattr(torch,dtype))
    frequencies = tuple(config['frequencies_hz'])
    batch = _positive_count(config.get('source_batch_size',1), 'source_batch_size')
    batch = min(batch,len(acquisition.sources_m))
    steps = _positive_count(config.get('steps',1), 'steps', allow_zero=True)
    preflight = config['preflight']
    if preflight.get('classification') != 'algebraic_smoke':
        raise ValueError('CLI currently requires classification=algebraic_smoke; physical validation is separate')
    dof = (grid.shape[0]+boundary.pml_cells*(1 if boundary.top=='free_surface' else 2)-2)*(grid.shape[1]+2*boundary.pml_cells-2)
    if dof > min(_positive_count(preflight['max_compact_dof'],'max_compact_dof'),10000):
        raise ValueError('preflight rejected compact dof; larger cases need a separate validated resource plan')
    rhs_count = len(frequencies)*len(acquisition.sources_m)
    if rhs_count > min(_positive_count(preflight['max_forward_rhs'],'max_forward_rhs'),32):
        raise ValueError('preflight rejected RHS count; broad sweeps require separate accuracy/resource validation')
    if steps > 10:
        raise ValueError('smoke preflight permits at most ten accepted optimizer steps')
    optimizer_config = dict(config.get('optimizer', {'name':'Adam','lr':.01}))
    name = optimizer_config.pop('name')
    if name not in ('Adam','SGD','LBFGS'):
        raise ValueError('CLI optimizer.name must be Adam, SGD or LBFGS; use native Python API for the full inventory')
    # Resolve the installed native default, rather than the planner's unrelated
    # history default. Adam/SGD have no LBFGS correction-pair history.
    history_size = (optimizer_config.get('history_size',
                    inspect.signature(torch.optim.LBFGS).parameters['history_size'].default)
                    if name == 'LBFGS' else 0)
    recording = dict(config.get('recording',{}))
    cadence = recording.pop('checkpoint_cadence',1)
    fields = tuple(recording.pop('fields',('velocity','slowness_squared')))
    archive = output/'accepted.h5'
    recording_policy = RecordingPolicy(archive,fields=fields,**recording)
    pool_slots = min(recording_policy.queue_capacity,
                     recording_policy.max_queue_bytes // recording_policy.chunk_bytes)
    recording_host_bytes = pool_slots*recording_policy.chunk_bytes
    recording_device_bytes = recording_host_bytes if torch.device(solver_config.device).type == 'cuda' else 0
    recording_buffer_bytes = recording_host_bytes+recording_device_bytes
    extra_workspace_bytes = _positive_count(preflight.get('extra_workspace_bytes',0),
                                            'extra_workspace_bytes',allow_zero=True)
    # This is an explicitly scoped complex128 matrix-free estimate, including
    # the configured Krylov basis, native LBFGS history, and archival pool.
    # It is not a strict process-memory envelope; excluded allocations are
    # serialized alongside the estimate so a passing screen is not a guarantee.
    plan = plan_stage(StageRequest((grid,), bounds, frequencies, len(acquisition.sources_m),
        len(acquisition.receivers_m), boundary=boundary, source_batch_sizes=(batch,),
        krylov_restart=solver_config.restart, optimizer_history_size=history_size,
        extra_workspace_bytes=recording_buffer_bytes+extra_workspace_bytes,
        max_phase_error=preflight.get('max_phase_error',.01)),
        memory_budget_bytes=solver_config.memory_budget_bytes)
    api = AcousticFWI(grid, acquisition, boundary, solver_config)
    real_dtype = torch.float64 if solver_config.dtype == torch.complex128 else torch.float32
    true = true.to(device=solver_config.device,dtype=real_dtype)
    initial = initial.to(device=solver_config.device,dtype=real_dtype)
    setup_time = time.perf_counter()-started
    forward_start = time.perf_counter()
    observed = api.simulate(true, frequencies, source_batch_size=batch)
    forward_seconds = time.perf_counter()-forward_start
    reports = [report.to_dict() for report in api.last_solve_reports]
    identity = {'parameterization': 'dense', 'dataset_hash': _hash_tensor(observed),
                'model_hash': _hash_tensor(initial), 'config_hash': _hash_json(config),
                'source_hash': acquisition.config_hash(), 'grid_hash': grid.config_hash()}
    objective = api.objective(observed, frequencies)
    model = VelocityParameters(initial, bounds)
    optimizer = getattr(torch.optim,name)(model.parameters(),**optimizer_config)
    runner = InversionRunner(objective,model,optimizer,source_batch_size=batch)
    if resume is not None:
        runner.accepted_steps = restore_checkpoint(load_checkpoint(resume,identity),model,optimizer)
    recording['metadata'] = {**identity, 'grid':grid.to_dict(), 'boundary':boundary.to_dict(),
        'acquisition': acquisition.to_dict(), 'config':config, 'source_ids':list(range(len(acquisition.sources_m))),
        'data_axes':['frequency','source','receiver'], 'provenance':provenance}
    gradient_start = time.perf_counter()
    with Recorder(RecordingPolicy(archive,fields=fields,**recording)) as recorder:
        adapter = AcceptedStateRecorder(recorder, checkpoint_path=output/'checkpoint.pt',config=identity,
                                       metadata=recording['metadata'],checkpoint_cadence=cadence)
        runner.accepted_callbacks = (adapter,)
        result = runner.run(steps)
    optimization_seconds = time.perf_counter()-gradient_start
    m = model.slowness_squared().detach().requires_grad_()
    gradient_start = time.perf_counter()
    for loss in objective.iter_data_losses(m,batch):
        loss.backward()
    gradient_seconds = time.perf_counter()-gradient_start
    if m.grad is None or not bool(torch.isfinite(m.grad).all()):
        raise FloatingPointError('smoke physical gradient failed')
    # Preserve observations and axes without pickled objects.
    import h5py
    with h5py.File(output/'observed.h5','x') as handle:
        ds = handle.create_dataset('data',data=observed.detach().cpu().numpy())
        ds.attrs['axes'] = json.dumps(['frequency','source','receiver'])
        ds.attrs['units'] = 'source_unit'
        ds.attrs['frequencies_hz'] = json.dumps(frequencies)
        ds.attrs['source_hash'] = acquisition.config_hash()
    source_x = sorted({position[1] for position in acquisition.sources_m})
    usable = grid.shape[1] if boundary.pml_cells else max(0,grid.shape[1]-2)
    manifest = {'schema_version':1,'torch_version':str(torch.__version__),'config':config,
        'identity':identity,'provenance':provenance,'grid':grid.to_dict(),'boundary':boundary.to_dict(),
        'preconditioner': None if api.preconditioner_config is None else api.preconditioner_config.to_dict(),
        'accepted_steps':runner.accepted_steps,'history':[asdict(value) for value in result.history],
        'accuracy':{'validation_status':plan.validation_status,'physical_accuracy_validated':False,
            'data_classification':'same_grid_algebraic_data','dispersion_scope':plan.dispersion.scope,
            'phase_error':plan.dispersion.max_phase_error,'phase_error_target':plan.dispersion.target_phase_error,
            'frequency_ceiling_hz':plan.dispersion.frequency_ceiling_hz,'dispersion_passed':plan.dispersion.passed,
            'pml_validation':'unrun_for_this_case','independent_discretization':'unrun',
            'true_residual_reports':reports},
        'source_coverage':{'source_count':len(acquisition.sources_m),'receiver_count':len(acquisition.receivers_m),
            'usable_horizontal_columns':usable,'distinct_source_x':len(source_x),
            'sampled_aperture_m':source_x[-1]-source_x[0],'usable_aperture_m':(usable-1)*grid.spacing_m[1],
            'marginal_information_benefit':'unmeasured','practical_source_limit':'one_per_usable_horizontal_column'},
        'resources':{'physical_cells':math.prod(grid.shape),'compact_dof':dof,
            'padded_cells':math.prod(api.operator.padded_shape),'forward_rhs':rhs_count,
            'setup_seconds':setup_time,'forward_all_rhs_seconds':forward_seconds,
            'optimization_seconds':optimization_seconds,'gradient_seconds':gradient_seconds,
            'physical_gradient_norm':float(torch.linalg.vector_norm(m.grad)),
            'hvp_seconds':None,'estimated_matrix_free_bytes':plan.memory.total_bytes,
            'memory_estimate_scope':plan.memory.scope,
            'memory_budget_is_process_limit':False,
            'memory_preflight':{'krylov_restart':plan.krylov_restart,
                'optimizer_history_size':history_size,
                'recording_host_buffer_bytes':recording_host_bytes,
                'recording_device_buffer_bytes':recording_device_bytes,
                'recording_buffer_bytes':recording_buffer_bytes,
                'caller_extra_workspace_bytes':extra_workspace_bytes,
                'total_extra_workspace_bytes':plan.memory.extra_workspace_bytes},
            'memory_estimate_exclusions':['Python/PyTorch runtime and allocator overhead',
                'direct LU factors and backend factorization workspace',
                'multigrid hierarchy storage beyond the generic operator reserve',
                'native optimizer trial rollback and checkpoint state snapshots/staging',
                'HDF5/compression caches, metadata and selection/index temporary buffers'],
            'direct_factorization_budget_scope':'retained_LU_only',
            'max_forward_report_bytes':max(report['memory_bytes'] for report in reports),
            'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'rss_scope':'whole_process_high_water_mark_including_imports',
            'archive_bytes':archive.stat().st_size,'recorder_peak_buffer_bytes':recorder.peak_buffer_bytes,
            'recording_comparison':'unrun','device':solver_config.device,'runpod_cost_usd':0.,
            'total_seconds':time.perf_counter()-started},
        'planned_unrun':config.get('planned_unrun',[]),
        'artifacts':{'accepted_archive':'accepted.h5','observed':'observed.h5',
                     'checkpoint':'checkpoint.pt' if (output/'checkpoint.pt').exists() else None}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return manifest
