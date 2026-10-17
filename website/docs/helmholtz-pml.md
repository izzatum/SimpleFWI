---
title: Acoustic Helmholtz solver with PML in PyTorch
description: How SimpleFWI solves the 2D frequency-domain acoustic Helmholtz equation with padded PML, an optional free surface, CPU sparse-direct and CUDA FGMRES solvers.
---

# Acoustic Helmholtz solver with PML

SimpleFWI solves the **2D constant-density acoustic Helmholtz equation** in
PyTorch, using a conservative second-order spatial discretization:

```text
(∇² + ω² m) p = q
m = 1 / v²,    ω = 2πf,    Fourier convention: exp(-iωt)
```

Here `v` is velocity in m/s, `m` is squared slowness in s²/m², `p` is the complex
wavefield and `q` is the source. The current API does not model elasticity,
variable density, 3D propagation or topography.

## How is PML implemented?

**SimpleFWI adds PML cells outside the physical grid and applies cubic complex
coordinate stretching inside that padding.** Padding and coordinate stretching
work together: the physical model stays on its declared node grid, and the
additional cells absorb outgoing waves. Outer pressure-boundary degrees of
freedom are eliminated. The code uses PML, not a replacement local absorbing
boundary condition.

`BoundaryConfig(top='pml', pml_cells=10)` requests PML on all four sides.
`BoundaryConfig(top='free_surface', pml_cells=10)` removes top padding and
imposes zero pressure at the physical top; side and bottom PML remain.
Sources and receivers cannot lie on eliminated zero-pressure nodes.

The velocity extension into PML participates consistently in physical-model
derivatives. PML thickness and damping still require calibration: a chosen cell
count alone does not establish a reflection-error bound. When refining a grid,
double the PML cell count if spacing halves to preserve physical width.

## CPU reference and experimental GPU solver

| Solver | Formulation | Current evidence |
| --- | --- | --- |
| CPU sparse direct | Assemble a sparse Helmholtz system; reuse bounded factor caches | Reference path for numerical comparisons |
| CPU/CUDA FGMRES | Matrix-free operator actions with a shifted-Helmholtz multigrid preconditioner | Ten CUDA tests passed on an NVIDIA A40; the iterative path remains experimental |

The solver checks the true unpreconditioned residual against explicit
tolerances. A failed solve raises before its gradient can be accepted. Requested
devices and precision are explicit; unavailable CUDA does not silently select
CPU. Matrix-free iterative actions do not imply that CPU direct factorization
or stored optimizer state consumes no memory.

The small recorded Marmousi continuation took 1.04 s on CPU direct and 13.08 s
on GPU FGMRES. Those single measurements establish no GPU speedup or
large-survey scaling. Higher-frequency and harder iterative cases can fail
convergence. Read the [measured validation limits](validation.md).

## Sources, receivers and adjoint gradients

`Grid2D` declares `(nz, nx)` nodes, `(dz, dx)` spacing and a first-node origin.
`Acquisition` uses `(z, x)` source and receiver coordinates in metres. Bilinear
receiver interpolation and its transpose scatter use identical weights. Source
amplitudes are normalized by cell area; recorded pressure is in numerical
`source_unit`, not calibrated pascals.

`AcousticFWI.simulate` returns complex sampled data in
`[frequency, source, receiver]` order. Implicit adjoint differentiation solves
the adjoint equation rather than retaining a differentiation graph through all
factorization or Krylov steps. Source batches limit live wavefields, while
stored observations, solver workspace and native optimizer state remain
separate memory costs.

## Does dispersion screening correct the physics automatically?

No. The accuracy screen checks grid spacing, frequency and minimum allowed
velocity against the discretization's phase-error criterion. It rejects unsafe
requested combinations; it does not secretly lower the frequency, change
precision or apply an unvalidated correction. Passing the screen does not prove
accuracy in a heterogeneous medium or calibrate PML reflections.

Before a physical accuracy study, compare grid refinement, PML thickness and
independent reference solutions. See the
[accuracy guide](https://github.com/izzatum/SimpleFWI/blob/main/guides/accuracy.md)
and [operator architecture](https://github.com/izzatum/SimpleFWI/blob/main/guides/architecture.md).

## Try the Marmousi wavefield tutorial

Open the [CPU/GPU Helmholtz notebook](https://github.com/izzatum/SimpleFWI/blob/main/notebooks/01_helmholtz_marmousi_cpu_gpu.ipynb).
It models the complete bundled Marmousi grid on CPU, plots wavefields and
residuals, then uses a smaller crop for bounded CPU/CUDA solver comparisons.
It explicitly identifies whether an actual CUDA device is available. The
low-frequency full-grid run and crop comparison do not establish calibrated
full-model accuracy, high-frequency performance or inversion recovery.
