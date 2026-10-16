# Tutorial notebooks

These tutorials use `simplefwi.fwi2d`, SI units, Python 3.12, and PyTorch 2.12.0.
Read them in order or open the topic you need. Saved figures and numerical
outputs make each notebook useful before running it yourself.

| Notebook | Contents | Default scope |
| --- | --- | --- |
| [01 — Helmholtz on CPU and GPU](01_helmholtz_marmousi_cpu_gpu.ipynb) | Marmousi geometry, PML, sampled data, complex wavefields, sparse-direct versus matrix-free solves | Full-model CPU demonstration plus a bounded Marmousi crop for matched device comparisons |
| [02 — FWI with native optimizers](02_fwi_marmousi_optimizers.ipynb) | Adam, RMSprop, L-BFGS, accepted steps, closure evaluations, losses and model comparisons | Fixed Marmousi crop and same-grid synthetic observations |
| [03 — Hessians and uncertainty](03_camembert_hessian_uncertainty.ipynb) | Exact and Gauss–Newton actions, residual curvature, proper-prior precision, local covariance and uncertainty figures | Camembert with a small active parameter window |

The first notebook explicitly reports whether CUDA ran. The notebooks never
create cloud resources. CPU and GPU comparisons use the same discrete problem;
an unavailable GPU is not replaced by CPU timing labeled as GPU timing.

## Install and launch

From the checkout, complete the [main installation](../README.md), activate its
virtual environment, and add the existing notebook extra:

```bash
python -m pip install -c requirements/fwi2d-py312.lock -e '.[fwi2d,notebooks]'
python -m jupyter lab notebooks/
```

Select the kernel from this environment. The setup cell prints the Python
executable, PyTorch version, device and precision. On a GPU machine, install the
official PyTorch 2.12.0 CUDA wheel appropriate for its driver before installing
the repository requirements. The legacy `gpu-cuda` extra is not the new PyTorch
solver's installation path.

Run **Restart Kernel and Run All**. Repository/data discovery works from the
checkout root or the `notebooks` directory. Keep the bundled `data/marmousi.mat`
file and Camembert fixture in their repository locations. All randomness uses
visible seeds, and model coordinates retain their physical origins.

For headless execution from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m jupyter nbconvert \
  --execute --to notebook --inplace --ExecutePreprocessor.timeout=300 \
  notebooks/01_helmholtz_marmousi_cpu_gpu.ipynb \
  notebooks/02_fwi_marmousi_optimizers.ipynb \
  notebooks/03_camembert_hessian_uncertainty.ipynb
```

This refreshes saved outputs on your hardware. In a restricted cloud workspace,
set `JUPYTER_DATA_DIR`, `JUPYTER_CONFIG_DIR`, `JUPYTER_RUNTIME_DIR`, `IPYTHONDIR`
and `MPLCONFIGDIR` to writable cache directories if their default locations are
read-only. No notebook server is needed for headless execution.

## Validation

All three default notebooks were executed top-to-bottom on CPU on 15 October
2026: **30 code cells, no execution errors, and 16 saved figures**. The same
headless command above was exercised independently. Scientific review checked
units, optimizer work, Hessian identities and covariance interpretation; visual
review checked native plots and rendered notebooks at desktop and narrow widths.
Open multi-panel figures at their native size or zoom when using a small screen.
The CPU CI job now executes these tutorials on future changes.

Notebook 01 subsequently executed its unchanged **11 code cells without errors**
on NVIDIA A40, driver 580.178.04, PyTorch 2.12.0+cu130 and CUDA 13.0 at source
revision `abbfe1f`. Its full Marmousi forward calculation remains a CPU run;
the bounded CUDA crop passed the residual and `<1e-7` waveform/data agreement
assertions. Table errors formatted as `0.0` are rounding, not exact-zero errors.
The [GPU execution summary](../benchmarks/fwi2d/results/runpod-notebook-01-summary.json)
records the code hash, kernel and numerical outcome.

The public notebooks retain their reviewed **CPU execution outputs and all 16
figures**. The [supplemental GPU notebook](../benchmarks/fwi2d/results/runpod-notebook-01.ipynb.gz)
is a gzip-compressed execution artifact with no embedded plots because the
headless backend did not capture its figures. It is separate evidence, not a
replacement or mixture of runs. Notebooks 02 and 03 remain CPU-only validation.
The [hardware record](../benchmarks/fwi2d/results/runpod-multiscale.json) also
contains ten passed library CUDA tests, a small solver benchmark and three-stage
Marmousi example. Selected SGD/Adam paths are covered; other optimizer/device
combinations and GPU speedup are not established.

The completed test pod was deleted and its absence verified; existing global
storage was preserved. Earlier
[SSH/Jupyter failures](../benchmarks/fwi2d/results/runpod-attempts.json) remain
recorded separately.

On a machine with actual CUDA, install the matching PyTorch wheel and run
notebook 01. It reports the executed device and performs its own numerical
comparison. Terminate/delete a rented GPU pod after use and preserve only the
storage you intend to retain. See the [bounded RunPod protocol](../benchmarks/fwi2d/runpod_protocol.md)
and [persistent-output guidance](../guides/outputs.md#persistent-cloud-storage).

## Interpret the results

- A small linear residual establishes solution of the discrete equation; it
  does not establish continuum accuracy. Dispersion and PML require separate
  physical checks.
- Optimizer comparisons use identical data and starting models. L-BFGS can
  evaluate several closures per accepted step, so compare computational work
  as well as iteration counts. Short same-grid inversions are usage examples,
  not evidence of full Marmousi reconstruction or an optimizer ranking.
- The exact Hessian can be indefinite away from a stationary minimum. A
  Gauss–Newton matrix becomes a covariance precision only after noise scaling,
  parameter coordinates, and a proper prior are specified. Displayed local
  intervals do not include model or discretization error.
- Explicit Jacobian/Hessian/covariance matrices are restricted to the small
  teaching window. Use operator actions for larger problems.

See [accuracy](../guides/accuracy.md), [uncertainty](../guides/uncertainty.md),
[output/restart](../guides/outputs.md), and the
[RunPod protocol](../benchmarks/fwi2d/runpod_protocol.md). The repository code and
these tutorials are distributed under the [MIT License](../LICENSE). Dataset
paths, parameters and provenance are recorded in the tutorials.
