"""Bounded full-domain CPU forward/gradient benchmark; never an accuracy claim.

Run from any cwd with the installed package and explicit dataset/output paths:
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_marmousi_cpu.py DATA.mat REPORT.json
Independent PML/heterogeneous accuracy and inversion recovery are not validated.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import torch
from simplefwi.fwi2d import (AcousticFWI, Acquisition, BoundaryConfig, Grid2D,
    SolverConfig, StageRequest, load_marmousi, nested_surface_acquisition, plan_stage)


def main(dataset, output):
    torch.set_num_threads(1)
    start = time.perf_counter()
    limit_seconds = 180.
    budget = 2*1024**3
    output = Path(output)
    report = {'schema_version':1,'status':'preflight','dataset_path':str(Path(dataset).resolve()),
        'dataset_sha256':hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        'input_velocity_unit':'m/s','torch_version':str(torch.__version__),'device':'cpu',
        'cpu_threads':torch.get_num_threads(),'budget_bytes':budget,'wall_cap_seconds':limit_seconds,
        'accuracy':{'physical_accuracy_validated':False,'data_classification':'same_grid_algebraic_data',
            'pml_validation':'unrun_for_full_Marmousi','independent_discretization':'unrun',
            'inversion_recovery':'unrun'},'cases':[],'peak_process_rss_bytes':0,'runpod_cost_usd':0.}
    def rss():
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1])*1024
        raise RuntimeError('Linux VmRSS unavailable')
    def save():
        output.parent.mkdir(parents=True,exist_ok=True)
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temporary.replace(output)
    stop = threading.Event()
    def watch():
        while not stop.wait(.1):
            report['peak_process_rss_bytes'] = max(report['peak_process_rss_bytes'],rss())
            elapsed = time.perf_counter()-start
            reason = 'rss_budget_exceeded' if report['peak_process_rss_bytes'] > budget else (
                     'wall_cap_exceeded' if elapsed > limit_seconds else None)
            if reason:
                report.update(status='stopped',stop_reason=reason,elapsed_seconds=elapsed)
                save()
                os._exit(3)
    watchdog = threading.Thread(target=watch,daemon=True)
    watchdog.start()
    try:
        grid,true = load_marmousi(dataset,velocity_unit='m/s')
        boundary = BoundaryConfig(pml_cells=32,sigma_max=60.)
        padded = tuple(n+64 for n in grid.shape)
        dof = (padded[0]-2)*(padded[1]-2)
        # Explicitly speculative fill estimate, separate from measured allocator
        # peaks. Allow 160 complex entries/DOF in LU and 3x transient reserve.
        factor_estimate = dof*160*16
        extra = factor_estimate*3+rss()
        plan = plan_stage(StageRequest((grid,),(1400.,6500.),(2.,),371,371,
            boundary=boundary,source_batch_sizes=(4,),extra_workspace_bytes=extra),
            memory_budget_bytes=budget)
        report['preflight'] = {'status':'spatial_and_resource_estimate_passed',
            'validation_status':plan.validation_status,'grid':grid.to_dict(),'boundary':boundary.to_dict(),
            'pml_thickness_m':800.,'padded_shape':padded,'compact_dof':dof,
            'phase_error':plan.dispersion.max_phase_error,'phase_error_target':.01,
            'frequency_ceiling_hz':plan.dispersion.frequency_ceiling_hz,
            'estimated_total_bytes':plan.memory.total_bytes,'assumed_LU_bytes':factor_estimate,
            'LU_estimate_scope':'heuristic160_complex_entries_per_dof_times3_transient_reserve',
            'memory_guarantee':False,'runtime_and_RSS_watchdog':True}
        if plan.memory.total_bytes > budget:
            raise RuntimeError('memory preflight exceeded budget')
        report['status']='running'
        save()
        bounds=(1400.,6500.)
        perturbation=.99
        for source_count in (8,371):
            # Exclude horizontal physical endpoint columns while preserving the
            # dyadic nesting of every interior source from the smaller set.
            full = nested_surface_acquisition(grid,boundary,source_count=source_count+2,
                source_depth_index=1,receiver_depth_index=2)
            interior = tuple(p for p in full.sources_m if grid.origin_m[1] < p[1] <
                             grid.origin_m[1]+(grid.shape[1]-1)*grid.spacing_m[1])
            receivers = full.receivers_m[1:-1]
            geometry=Acquisition(interior,receivers)
            api=AcousticFWI(grid,geometry,boundary,SolverConfig(memory_budget_bytes=budget))
            case={'source_count':len(interior),'receiver_count':len(receivers),'batch_size':4,
                  'usable_source_columns':371,'source_hash':geometry.config_hash(),
                  'source_positions_m':geometry.to_dict()['sources_m'],
                  'receiver_spacing_m':25.,'source_depth_m':25.,'receiver_depth_m':50.,
                  'source_coverage_fraction':len(interior)/371,
                  'physical_source_aperture_m':max(p[1] for p in interior)-min(p[1] for p in interior),
                  'marginal_information_benefit':'unmeasured','status':'forward_running'}
            report['cases'].append(case)
            save()
            forward_start=time.perf_counter()
            data=api.simulate(true,(2.,),source_batch_size=4)
            case['forward_all_rhs_seconds']=time.perf_counter()-forward_start
            case['forward_LU_factorizations']=api.solver.factorization_count
            case['forward_max_relative_residual']=max(r.relative_residual for r in api.last_solve_reports)
            case['forward_max_absolute_residual']=max(r.absolute_residual for r in api.last_solve_reports)
            case['max_forward_report_bytes']=max(r.memory_bytes for r in api.last_solve_reports)
            case['status']='gradient_running'
            save()
            objective=api.objective(data,(2.,))
            m=(true*perturbation).reciprocal().square().requires_grad_()
            gradient_start=time.perf_counter()
            loss_value=0.
            for loss in objective.iter_data_losses(m,source_batch_size=4):
                loss_value+=float(loss.detach())
                loss.backward()
            case['gradient_all_rhs_seconds']=time.perf_counter()-gradient_start
            case['total_LU_factorizations']=api.solver.factorization_count
            case['synthetic_initial_velocity_scale']=perturbation
            case['data_loss']=loss_value
            case['physical_gradient_norm']=float(torch.linalg.vector_norm(m.grad))
            case['finite_gradient']=bool(torch.isfinite(m.grad).all())
            if not case['finite_gradient']:
                raise FloatingPointError('nonfinite full-domain physical gradient')
            case['status']='completed'
            case['retained_LU_bytes']=api.solver.cache_bytes
            api.solver.clear_cache()
            report['peak_process_rss_bytes']=max(report['peak_process_rss_bytes'],rss())
            save()
        report['status']='completed'
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-start
        report['peak_process_rss_bytes']=max(report['peak_process_rss_bytes'],rss())
        stop.set()
        save()
    print(output)


if __name__=='__main__':
    if len(sys.argv)!=3:
        raise SystemExit('usage: run_marmousi_cpu.py DATA.mat REPORT.json')
    main(*sys.argv[1:])
