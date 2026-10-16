"""Small Marmousi continuation experiment; explicit CPU or actual CUDA execution.

This uses fine-generated synthetic data and a cross-grid coarse stage. Its short
PML is not independently calibrated; it is not a full Marmousi reconstruction
or a GPU speedup benchmark. The initial model lives on the coarse grid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from simplefwi.fwi2d import (
    AcousticFWI, Acquisition, BoundaryConfig, Grid2D, MultiscaleInversion,
    MultiscaleStage, SolverConfig, load_marmousi,
)


def run(*, device='cpu', quick=False, steps=3):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    real, complex_dtype = torch.float64, torch.complex128
    dataset_path = Path(__file__).resolve().parents[2]/'data'/'marmousi.mat'
    original_grid, velocity = load_marmousi(dataset_path, velocity_unit='m/s')
    provenance = dict(path='data/marmousi.mat', velocity_unit='m/s',
        sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest())
    # Preserve dataset coordinates, including its nonzero horizontal origin.
    start_z, start_x = 20, 120
    fine_shape = (9, 13) if quick else (17, 25)
    truth = velocity[start_z:start_z+fine_shape[0], start_x:start_x+fine_shape[1]].to(device)
    fine_grid = Grid2D(fine_shape, original_grid.spacing_m,
        (original_grid.origin_m[0]+start_z*original_grid.spacing_m[0],
         original_grid.origin_m[1]+start_x*original_grid.spacing_m[1]))
    coarse_grid = Grid2D(tuple((n+1)//2 for n in fine_shape),
        tuple(2*h for h in fine_grid.spacing_m), fine_grid.origin_m)
    z0, x0 = fine_grid.origin_m
    extent_x = (fine_grid.shape[1]-1)*fine_grid.spacing_m[1]
    acquisition = Acquisition(
        sources_m=((z0+50., x0+50.), (z0+50., x0+extent_x-50.)),
        receivers_m=tuple((z0+50., x0+float(x)) for x in
                         torch.linspace(25., extent_x-25., 7 if quick else 11,
                                        dtype=torch.float64)),
    )
    p = 2 if quick else 4
    coarse_boundary = BoundaryConfig(pml_cells=p, sigma_max=60.)
    fine_boundary = BoundaryConfig(pml_cells=2*p, sigma_max=60.)
    frequencies = (2., 3.)
    config = SolverConfig(method='direct' if device == 'cpu' else 'fgmres',
        device=device, dtype=complex_dtype, rtol=1e-11, atol=1e-14,
        restart=40, maxiter=240, memory_budget_bytes=512*1024**2)
    # Generate one fixed fine-grid observation tensor, shared by every stage.
    simulator = AcousticFWI(fine_grid, acquisition, fine_boundary, config)
    observed = simulator.simulate(truth, frequencies, source_batch_size=1)
    # Explicit global scalar weight keeps this example's step size practical.
    # The library itself never introduces a data/frequency-count normalization.
    weight = float(observed.norm().reciprocal())
    initial = torch.full(coarse_grid.shape, float(truth.mean()), dtype=real, device=device)
    stages = (
        MultiscaleStage(coarse_grid, (2.,), steps, coarse_boundary),
        MultiscaleStage(fine_grid, (2.,), steps, fine_boundary),
        MultiscaleStage(fine_grid, frequencies, steps, fine_boundary),
    )
    inversion = MultiscaleInversion(stages, acquisition, observed, frequencies,
        initial, (1400., 5500.), config,
        lambda model, stage: torch.optim.Adam(model.parameters(), lr=.03),
        weights=weight)
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = inversion.run()
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    stage_reports = []
    for item in result.stages:
        stage_reports.append(dict(index=item.index, shape=list(item.stage.grid.shape),
            spacing_m=list(item.stage.grid.spacing_m),
            frequencies_hz=list(item.stage.frequencies_hz),
            initial_loss=item.initial_loss,
            accepted_losses=[step.loss for step in item.history],
            closure_evaluations=sum(step.closure_evaluations for step in item.history),
            pml_widths_m=list(item.pml_widths_m),
            max_phase_error=item.max_phase_error,
            transfer_bound_corrections=item.transfer_bound_corrections,
            max_transfer_bound_correction_m_s=item.max_transfer_bound_correction_m_s))
    assert result.velocity.device.type == device
    assert result.velocity.dtype == real and torch.isfinite(result.velocity).all()
    assert result.grid == fine_grid and result.velocity.grad_fn is None
    return dict(schema_version=1, experiment='marmousi_multiscale_crop',
        validation_scope='synthetic integration; short PML not independently calibrated',
        device=device, torch_version=torch.__version__, cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name() if device == 'cuda' else None,
        dtype=str(complex_dtype), dataset=provenance,
        crop_velocity_sha256=hashlib.sha256(truth.cpu().contiguous().numpy().tobytes()).hexdigest(),
        origin_m=list(fine_grid.origin_m), source_count=len(acquisition.sources_m),
        receiver_count=len(acquisition.receivers_m), stages=stage_reports,
        inversion_seconds=elapsed,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
        final_velocity_min_m_s=float(result.velocity.min()),
        final_velocity_max_m_s=float(result.velocity.max()),
        final_velocity_rmse_m_s=float((result.velocity-truth).square().mean().sqrt()),
        solver_config=config.to_dict(), explicit_data_weight=weight,
        gpu_speedup_claim=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--quick', action='store_true', help='small algebraic integration crop')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    result = run(device=args.device, quick=args.quick, steps=args.steps)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
