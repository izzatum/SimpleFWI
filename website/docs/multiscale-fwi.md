---
title: Multi-resolution and multi-frequency FWI in PyTorch
description: Run coarse-to-fine and low-to-high frequency FWI with native PyTorch optimizers, physical squared-slowness transfer and a reproducible Marmousi example.
---

# Multi-resolution and multi-frequency FWI

**Multi-resolution FWI changes the model grid; multi-frequency continuation
changes the observations included in the objective.** SimpleFWI's
`MultiscaleInversion` schedules either or both. A typical sequence begins with
a coarse, low-frequency problem, refines the physical model and then introduces
additional frequencies.

## Run the three-stage Marmousi example

From an [installed checkout](index.md#install-the-validated-cpu-environment):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/fwi2d/multiscale_marmousi.py \
  --device cpu --quick --output /tmp/multiscale-cpu.json
```

The quick schedule uses a small crop of the bundled Marmousi data:

| Stage | Physical grid | Frequencies | Optimization |
| --- | --- | --- | --- |
| 1 | 5 × 7 nodes, 50 m spacing | 2 Hz | Three Adam steps |
| 2 | 9 × 13 nodes, 25 m spacing | 2 Hz | Three fresh Adam steps |
| 3 | Same 9 × 13 grid | 2 and 3 Hz jointly | Three fresh Adam steps |

For CUDA, install the matching CUDA PyTorch wheel and require an actual device:

```bash
python -c "import torch; assert torch.cuda.is_available(), 'CUDA device required'"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python examples/fwi2d/multiscale_marmousi.py \
  --device cuda --quick --output /tmp/multiscale-cuda.json
```

The JSON output records model/data hashes, stage losses, physical PML widths,
timing and device memory. The example uses synthetic fine-grid observations
and short, uncalibrated PML. It is an integration experiment, not demonstrated
full-model recovery.

## What happens between stages?

1. **Validate the full schedule.** Stages must preserve physical node extent,
   origin, acquisition coordinates and top-boundary type. Refinement is
   coarse-to-fine; the highest frequency cannot decrease. Dispersion is
   screened at the minimum declared velocity before optimization begins.
2. **Transfer physical squared slowness.** The accepted `m = 1 / velocity**2`
   is interpolated with physical nodes aligned. Same-grid stages preserve
   accepted velocity values exactly. Observations are selected by frequency;
   measured data are not spatially interpolated.
3. **Create fresh stage state.** The model, solver, optimizer and scheduler
   belong to that stage. Adam moments, momentum and L-BFGS history do not cross
   grid or objective transitions.
4. **Keep memory bounded by declared operations.** Retain scalar stage history
   and the final runner, clear obsolete solver caches, and process source
   batches sequentially. Observation copies and optimizer history still
   contribute to memory use.

Prepare the first coarse initial model explicitly. The workflow rejects
coarsening and does not silently decimate a fine model. Halving spacing requires
doubling PML cells to preserve physical width. Refinement creates computational
resolution, not new geological information.

## Use native PyTorch optimizers

Each stage accepts an optimizer factory, for example:

```python
import torch

optimizer_factory = lambda model, stage: torch.optim.Adam(
    model.parameters(), lr=0.03
)
```

Use this factory in the complete
[stage-declaration example](https://github.com/izzatum/SimpleFWI/blob/main/guides/multiscale.md#declare-the-stages).
The inversion runner supplies the objective and gradients; native `torch.optim`
controls updates. Fifteen CPU optimizer recipes are checked, including Adam,
RMSprop, L-BFGS, Muon and SparseAdam with their documented shape and gradient
constraints. That does not establish every optimizer configuration on CUDA.
The [optimizer guide](https://github.com/izzatum/SimpleFWI/blob/main/guides/optimizers.md)
states supported contracts and examples.

Optional factories declare masks, fixed physical priors and stage-local
recorders. Use separate output paths for each stage and close writers explicitly.
The final runner supports [curvature analysis](hessian-uncertainty.md) at the
final objective; its Hessian is not a derivative through the entire optimization
trajectory.

## Interpret a continuation result

Adding frequencies changes the objective. Compare initial and final loss
**within each stage**, not raw losses across different bands. In the recorded
CPU/A40 comparison, every stage's final loss was lower than its initial loss,
but final model RMSE increased from 33.41 to 34.84 m/s. Individual Adam steps
also need not decrease monotonically. Lower data misfit alone does not establish
better geological recovery.

See [validation and hardware evidence](validation.md) for the exact revision,
CPU/GPU parity and timing. The
[full multiscale guide](https://github.com/izzatum/SimpleFWI/blob/main/guides/multiscale.md)
documents frequency weights, transfer roundoff, priors, masks, schedulers and
failure handling.
