---
title: 2D full-waveform inversion in PyTorch
description: SimpleFWI is an MIT-licensed Python library for 2D frequency-domain acoustic FWI in PyTorch, with PML, multiscale inversion and Hessian actions.
---

# 2D full-waveform inversion in PyTorch

**SimpleFWI is a Python research library for 2D, constant-density acoustic
full-waveform inversion (FWI) in PyTorch.** It models seismic data in the
frequency domain with a Helmholtz solver, computes implicit adjoint gradients,
and updates bounded velocity models with native PyTorch optimizers. The code is
[MIT licensed](https://github.com/izzatum/SimpleFWI/blob/main/LICENSE).

Start with the CPU sparse-direct reference solver. An experimental matrix-free
CUDA FGMRES solver is available, with a recorded NVIDIA A40 validation run.
The current examples establish small-problem integration and numerical parity;
large-scale, high-frequency Marmousi recovery and GPU speedup remain unverified.

## What can you do with SimpleFWI?

| Task | Implementation and next step |
| --- | --- |
| Model acoustic wavefields | [Helmholtz equation with padded PML](helmholtz-pml.md), optional pressure-release free surface |
| Invert several frequencies and grid resolutions | [Coarse-to-fine and low-to-high continuation](multiscale-fwi.md), physical squared-slowness transfer |
| Choose an optimizer | Native `torch.optim` recipes, including Adam, RMSprop and L-BFGS; [15 CPU recipes checked](https://github.com/izzatum/SimpleFWI/blob/main/guides/optimizers.md) |
| Examine local uncertainty | [Exact and Gauss–Newton Hessian-vector products](hessian-uncertainty.md), proper-prior local covariance actions |
| Record and reproduce a run | [Accepted-state HDF5 recording and restart checkpoints](https://github.com/izzatum/SimpleFWI/blob/main/guides/outputs.md), explicit configuration and dataset hashes |

![Physical 2D grid with sources, receivers, circular velocity inclusion, added PML padding and an optional pressure-release top boundary](assets/fwi-workflow-domain.svg)

The illustration shows the complete
[README example geometry](https://github.com/izzatum/SimpleFWI#quickstart).
The workflow is **grid and acquisition → Helmholtz solve → receiver data →
weighted misfit → adjoint gradient → optimizer update**. Repeat the solves at
the selected frequencies and model resolutions. Curvature and recording are
explicit optional operations.

## Install the validated CPU environment

Use Python 3.12 and PyTorch 2.12.0. From a fresh checkout:

```bash
git clone https://github.com/izzatum/SimpleFWI.git
cd SimpleFWI
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/fwi2d-py312.lock
python -m pip install --no-deps -e '.[fwi2d]'
```

For NVIDIA CUDA, create a separate environment and select the matching official
PyTorch 2.12.0 CUDA wheel before installing the same lock file. The legacy
`gpu-cuda` extra is not the installation path for the PyTorch solver. Follow the
[installation guide](https://github.com/izzatum/SimpleFWI/blob/main/guides/quickstart.md).

## Run a small frequency-domain FWI example

From the installed checkout, use a new output path:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m simplefwi.fwi2d --output /tmp/simplefwi-first-run
```

This runs a small Camembert integration example. Inspect the output manifest's
`accuracy` classification before interpreting the model. For a complete Python
example using `AcousticFWI`, `VelocityParameters` and `InversionRunner`, see the
[executable README quickstart](https://github.com/izzatum/SimpleFWI#quickstart).

Models use SI units: metres, metres/second and Hz. Model arrays are **[z, x]**;
observations are **[frequency, source, receiver]**. Defaults are float64 model
parameters, complex128 wavefields and the Fourier convention `exp(-i omega t)`.

## Learn with executable notebooks

1. [Helmholtz solver on CPU and GPU: Marmousi](https://github.com/izzatum/SimpleFWI/blob/main/notebooks/01_helmholtz_marmousi_cpu_gpu.ipynb)
   shows wavefields, residuals and explicit device comparisons.
2. [FWI with Adam, RMSprop and L-BFGS: Marmousi](https://github.com/izzatum/SimpleFWI/blob/main/notebooks/02_fwi_marmousi_optimizers.ipynb)
   compares loss histories and measured computational work.
3. [Hessian and Gauss–Newton uncertainty: Camembert](https://github.com/izzatum/SimpleFWI/blob/main/notebooks/03_camembert_hessian_uncertainty.ipynb)
   compares local curvature and proper-prior covariance on a small active window.

All three default notebooks have recorded CPU executions; the Helmholtz notebook
also passed a separate bounded A40 execution. Read the
[validation record and FAQ](validation.md) for exact scope, numerical evidence
and known limitations.

## Cite and reproduce a result

Reference [izzatum/SimpleFWI](https://github.com/izzatum/SimpleFWI), retain the MIT
license notice, and record the Git revision, configuration, data provenance,
device, precision and solver tolerances used. No associated publication or DOI
is claimed. For questions or reproducible bug reports, use
[GitHub issues](https://github.com/izzatum/SimpleFWI/issues).
