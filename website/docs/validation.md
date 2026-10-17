---
title: SimpleFWI validation, GPU evidence and FAQ
description: Inspect 463 CPU tests and 10 NVIDIA A40 CUDA tests for SimpleFWI, small Marmousi CPU/GPU comparisons, supported physics and reproducibility limits.
---

# Validation, GPU evidence and FAQ

SimpleFWI is research software with explicit numerical evidence and limits.
The recorded suite at source revision
[`abbfe1f`](https://github.com/izzatum/SimpleFWI/commit/abbfe1fa7102580a725990e8de3a241c4874b354)
passed **463 CPU tests and 10 separate CUDA tests on an NVIDIA A40**. These
counts describe the `tests/fwi2d` selection at that revision, not a blanket
certification of the legacy repository, all hardware or full-scale recovery.

## Reproducible validation records

| Evidence | Result and scope |
| --- | --- |
| [CPU manifest](https://github.com/izzatum/SimpleFWI/blob/main/benchmarks/fwi2d/results/multiscale-cpu.json) | 463 passed; zero failures, errors or skips; 10 CUDA tests deselected |
| [A40 hardware manifest](https://github.com/izzatum/SimpleFWI/blob/main/benchmarks/fwi2d/results/runpod-multiscale.json) | 10 CUDA tests passed; zero failures, errors or skips; PyTorch 2.12.0+cu130, CUDA 13.0 |
| [CPU Marmousi crop](https://github.com/izzatum/SimpleFWI/blob/main/benchmarks/fwi2d/results/runpod-multiscale-cpu.json) and [CUDA crop](https://github.com/izzatum/SimpleFWI/blob/main/benchmarks/fwi2d/results/runpod-multiscale-cuda.json) | Three continuation stages; six initial/final losses agree within 6.22e-10 relative error |
| [Tutorial execution record](https://github.com/izzatum/SimpleFWI/blob/main/notebooks/README.md) | Three default CPU notebooks, 30 code cells, 16 saved figures; separate bounded CUDA execution of the Helmholtz notebook |

CPU checks include adjoints and second derivatives, physical fixtures, all
15 optimizer recipes, multiscale transitions, recording and restart. CUDA checks
include waveform and physical-gradient comparisons, small inversion,
PML/free-surface multiscale parity and bounded recording. The hardware record
does not certify every native optimizer or all precisions on GPU.

Reproduce the selections in the appropriate installed environment:

```bash
# CPU reference suite
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest tests/fwi2d -m 'not cuda' -q

# Actual CUDA hardware required; skipped tests are not GPU evidence
python -c "import torch; assert torch.cuda.is_available(), 'CUDA device required'"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest tests/fwi2d -m cuda -q -rs
```

## What the small Marmousi experiment shows

Every continuation stage ended below its initial data loss. However, final
velocity RMSE was **34.8429 m/s**, compared with **33.4132 m/s** for the initial
constant truth-mean model. Improved data fit did not improve model error in
this short experiment.

The matched inversion took **1.03665 s on CPU direct and 13.07533 s on GPU
FGMRES**, with **17,673,728 peak allocated CUDA bytes**. These are single
measurements on a small crop with short, uncalibrated PML. They establish
numerical workflow parity, not GPU speedup, large-survey memory scaling,
independent continuum accuracy or high-frequency full Marmousi recovery.

The [full validation guide](https://github.com/izzatum/SimpleFWI/blob/main/guides/validation.md)
also records harder iterative convergence failures and pre-existing legacy
whole-repository pytest collection failures. A low linear residual or a passed
dispersion screen alone does not establish physical accuracy.

## Frequently asked questions

### What is SimpleFWI?

SimpleFWI is an MIT-licensed Python research library for 2D constant-density
acoustic frequency-domain full-waveform inversion in PyTorch. Its API combines
Helmholtz modeling, adjoint gradients, bounded native optimization and local
curvature analysis.

### Can SimpleFWI run full-waveform inversion on a GPU?

Yes. The experimental CUDA FGMRES path passed ten numerical tests on an NVIDIA
A40, including small inversion and multiscale comparisons. CPU sparse-direct
solves remain the reference default. Large-scale high-frequency performance
and GPU speedup are unestablished.

### Does SimpleFWI use PML and support a free surface?

Yes. It pads the physical grid with stretched-coordinate PML. An optional flat
pressure-release free surface replaces the top PML; side and bottom PML remain.
PML thickness and damping require calibration for the chosen grid and frequency.
See [Helmholtz and PML](helmholtz-pml.md).

### How do multi-resolution and multi-frequency FWI differ?

Multi-resolution changes the grid; multi-frequency continuation changes the
frequencies in the objective. SimpleFWI schedules both, transfers physical
squared slowness on refinement and creates fresh optimizer state per stage.
See the [Marmousi continuation example](multiscale-fwi.md).

### Can I use PyTorch optimizers and compute a Hessian?

Yes. The runner integrates native PyTorch optimizer recipes; fifteen CPU recipes
are tested with their documented constraints. Exact and Gauss–Newton
Hessian-vector products avoid dense Hessian assembly. Local uncertainty also
requires noise scaling, declared parameter coordinates and a proper prior.
See [Hessian and uncertainty](hessian-uncertainty.md).

### What wave physics and problem sizes are supported?

The current API is 2D, constant-density and acoustic, with a flat optional free
surface. It does not implement 3D, elastic or variable-density inversion. The
recorded GPU experiments use small fixtures and crops; full-model high-frequency
scalability remains unverified.

### How should I cite or report a SimpleFWI result?

Attribute [the repository and contributors](https://github.com/izzatum/SimpleFWI),
record the exact revision and numerical configuration, and retain the MIT
license notice. Include dataset provenance, noise/weight assumptions, model
coordinates, device, precision and solver tolerances. No associated publication
or DOI is claimed by these pages.
