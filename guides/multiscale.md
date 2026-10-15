# Multi-resolution and multi-frequency FWI

`AcousticFWI.objective` can already combine several frequencies in one objective.
`MultiscaleInversion` adds **continuation**: solve an easier low-frequency,
coarse-grid problem, transfer the recovered model, then introduce finer grids
and higher frequencies. Both changes can be scheduled independently.

```mermaid
flowchart LR
    A[Coarse grid: 2 Hz] -->|refine physical squared slowness| B[Fine grid: 2 Hz]
    B -->|add observations; reset optimizer| C[Fine grid: 2 and 3 Hz]
    C --> D[Final model and stage-local Hessian]
```

## Declare the stages

Supply the initial velocity on the **first/coarsest grid**. Every grid must have
the same origin and physical node extent: `(shape - 1) * spacing`. For example,
17 by 21 nodes at 50 m refine to 33 by 41 nodes at 25 m. Acquisition locations
remain in metres and are sampled anew on each stage grid.

```python
import torch
from simplefwi.fwi2d import (
    BoundaryConfig, Grid2D, MultiscaleInversion, MultiscaleStage, SolverConfig,
)

coarse = Grid2D((17, 21), (50., 50.))
fine = Grid2D((33, 41), (25., 25.))
coarse_pml = BoundaryConfig(top='pml', pml_cells=10)
fine_pml = BoundaryConfig(top='pml', pml_cells=20)
stages = (
    MultiscaleStage(coarse, (2.,), 5, coarse_pml),
    MultiscaleStage(fine, (2.,), 5, fine_pml),
    MultiscaleStage(fine, (2., 3.), 10, fine_pml),
)

# acquisition: one fixed Acquisition with coordinates in metres.
# observed: complex128 tensor [2 frequencies, sources, receivers].
# initial_velocity: float64 tensor [17, 21] in m/s, inside the bounds.
inversion = MultiscaleInversion(
    stages, acquisition, observed, (2., 3.), initial_velocity,
    (1400., 5500.), SolverConfig(),
    lambda model, stage: torch.optim.Adam(model.parameters(), lr=.03),
)
result = inversion.run()
velocity = result.velocity             # final grid, detached tensor
for stage_result in result.stages:
    print(stage_result.stage.frequencies_hz, stage_result.initial_loss,
          [step.loss for step in stage_result.history])
```

Use `(2.,)`, then `(3.,)` for single-frequency continuation; `(2.,)`, then
`(2., 3.)` for cumulative continuation; or explicit tuples for frequency bands.
The highest frequency cannot decrease, but cumulative bands can keep lower
frequencies. The master observations may be in any frequency order; each stage
selects the exact corresponding frequency rows and weights. Measurements are
never spatially interpolated. Weights broadcast over the **master**
`[frequency, source, receiver]` shape before stage selection.

The objective remains the sum of weighted residual squares plus one physical
regularizer. Adding frequencies changes the objective, so compare loss before
and after optimization **within a stage**, not across different stages.

## Transitions, accuracy and memory

The runner interpolates `m = 1 / velocity**2`, not velocity or unconstrained
optimizer parameters. Bilinear interpolation aligns the original physical
nodes, preserves constant and affine `m`, and runs on the selected device.
Same-grid stages copy the accepted model. Coarsening is rejected: prepare a
physically smoothed coarse initial model explicitly rather than decimating a
fine model and silently aliasing its structure. Refinement adds computational
resolution; it creates no new geological information.

Same-grid transitions preserve accepted velocity values exactly, including
frozen cells at a bound. After refinement, an outward excursion of at most
four dtype epsilons times that bound is corrected to the exact bound and counted
in `transfer_bound_corrections` and `max_transfer_bound_correction_m_s`.
Larger excursions fail. This handles floating-point inversion roundoff; it
never moves an active endpoint inward to make an otherwise invalid model pass.

Each stage uses a fresh bounded model, operator, solver and native optimizer.
Momentum, Adam moments, L-BFGS history and schedulers do not cross stage
boundaries. The optimizer factory can call any compatible native PyTorch recipe;
there is no optimizer-specific continuation algorithm. Active values at exactly
a velocity bound are rejected because the sigmoid inverse would be infinite;
the runner does not silently clip them into the interior.

PML is specified per stage. Halving the grid spacing requires doubling
`pml_cells` to preserve its physical width. The runner rejects shrinking either
physical width or switching the top boundary type. Use `top='free_surface'` at
all stages for a pressure-release top with side/bottom PML. No sources or
receivers may lie on pressure-zero top nodes. Wider PML still requires physical
calibration; preserving width does not prove reflection accuracy.

All stages undergo dispersion screening at the **minimum allowed velocity**
before optimization. A failed stage is rejected without changing requested
frequencies, tolerances or precision. Screening is not a heterogeneous-medium
accuracy guarantee, a PML calibration, or a convergence guarantee. The existing
experimental FGMRES solver still enforces its true residual tolerances.

Only scalar stage histories and the final model are retained. Source batches
remain sequential, and obsolete solver caches are cleared. The master data and
current stage's frozen objective copy still occupy memory. Native optimizer
history and rollback snapshots also contribute to memory consumption.

## Masks, priors, recording and curvature

Optional factories receive the immutable stage:

- `model_factory(velocity, stage)` returns `VelocityParameters` with an explicit
  target-grid active mask or sparse embedding. Preserve the supplied physical
  velocity, precision, device and bounds. Masks are not interpolated.
- `regularizer_factory(stage)` returns a physical regularizer built on that
  stage grid, with a fixed user-chosen prior. Transfer the original prior when
  needed; do not reset the prior to each recovered stage model.
- `configure_runner(index, stage, runner)` attaches a fresh scheduler, recorder
  or accepted-state callbacks before that stage runs. Use distinct archive
  paths such as `stage-000`, `stage-001`: accepted-step counters restart at one.
  The scheduler must own the current stage optimizer and have no prior history;
  the hook cannot replace the model, objective, optimizer or source batch size.
  Follow the existing [bounded recording guidance](outputs.md), close owned
  writers in a `finally` block, and avoid retaining runner objects in callbacks.

After `run`, `inversion.runner.objective` and `inversion.runner.model` are the
final-stage objective/model and work with the existing
[exact and Gauss–Newton curvature APIs](uncertainty.md). The Hessian describes
that final objective in its declared coordinates; it is not a derivative through
the entire optimization trajectory. Instantiate a new continuation object for a
new run; the runner prevents accidentally repeating a partially completed run.

## Runnable Marmousi example

From an installed checkout:

```bash
python examples/fwi2d/multiscale_marmousi.py --device cpu --quick --output /tmp/multiscale-cpu.json
python examples/fwi2d/multiscale_marmousi.py --device cuda --quick --output /tmp/multiscale-cuda.json
```

The example uses the bundled Marmousi dataset with explicit SI units and hashes,
then runs a coarse 2 Hz stage, fine 2 Hz stage and fine joint 2/3 Hz stage. It
reports each stage's losses, closure count, physical PML widths, dispersion
screening, timing and actual CUDA memory when available. `--device cuda`
requires a CUDA-enabled PyTorch installation and actual hardware; it fails
explicitly if either is unavailable. The default larger crop is still small.

This demonstration uses a short, uncalibrated PML and synthetic fine-grid
observations. The coarse-stage residual includes cross-grid discretization
mismatch; it is not an isolated estimate of that error. The example does not
establish full Marmousi recovery, high-frequency
scalability, independent continuum accuracy, or GPU speedup.

## Recorded CPU and CUDA validation

The [CPU validation artifact](../benchmarks/fwi2d/results/multiscale-cpu.json)
records **463 passed, zero failures/errors/skips, and 10 CUDA tests deselected**
at source revision `abbfe1f`, using:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/fwi2d -m 'not cuda' -q
```

The suite includes 96 multiscale CPU cases: physical-m transfer, frequency and
weight selection, independent manual-stage comparisons with Adam/LBFGS,
free-surface/PML geometry, fresh optimizer/scheduler state, masks, recording
lifecycle, curvature and failure cleanup. The same artifact contains executed
quick and default CPU Marmousi crops, refining 5×7 to 9×13 and 9×13 to 17×25
nodes respectively. Both finish each of their three stages below that stage's
initial loss; individual Adam steps need not decrease monotonically. These small
synthetic runs verify the continuation workflow, with the physical and scaling
limits stated above.

The separate [A40 hardware record](../benchmarks/fwi2d/results/runpod-multiscale.json)
at the same source revision reports **10 CUDA tests passed, zero failures/errors/
skips**, using PyTorch 2.12.0+cu130, CUDA 13.0 and driver 580.178.04. Two of those
tests execute coarse-to-fine Adam continuation with PML and free-surface
boundaries. They compare resident float64/complex128 waveforms and physical-m
gradients against CPU direct at a relative tolerance of `1e-7`, as well as stage
losses and final models. This is evidence for these small discrete fixtures,
not every optimizer, precision or acquisition.

The executed [GPU Marmousi crop](../benchmarks/fwi2d/results/runpod-multiscale-cuda.json)
matches its [CPU direct run](../benchmarks/fwi2d/results/runpod-multiscale-cpu.json):
the maximum relative difference across all six initial/final stage losses is
`6.22e-10`. GPU losses were:

| Stage | Initial loss | Final loss |
| --- | ---: | ---: |
| Coarse 2 Hz | 6.80276434e-4 | 4.74044842e-4 |
| Fine 2 Hz | 1.03691798e-4 | 1.50567289e-6 |
| Fine 2/3 Hz | 7.05500578e-6 | 6.55973156e-6 |

Each final loss is below its stage's initial loss, but the first Adam step of
stage three increases loss. Final velocity RMSE is **34.8429 m/s**, compared with
**33.4132 m/s** for the initial constant truth-mean model: reducing the data
objective did not improve model RMSE in this short experiment. This is workflow
and CPU/GPU numerical parity evidence, not a recovery result.

The matched inversion took **1.03665 s on CPU direct and 13.07533 s on GPU
FGMRES**, with 17,673,728 peak allocated CUDA bytes. These are single measurements
on a small crop with short, uncalibrated PML; they show no GPU speedup and establish
neither large-survey memory scaling nor high-frequency recovery. The separate GPU
solver benchmark and notebook 01's bounded CUDA comparison also passed, as
recorded in the hardware manifest. Historical CPU results retain their original
scope.
