# Implementation status — 15 October 2026

The new `simplefwi.fwi2d` workflow is implemented alongside the legacy API.
CPU numerical validation, ten selected CUDA checks, a small GPU solver benchmark,
Marmousi continuation example and notebook 01's bounded CUDA branch have passed.
High-frequency Marmousi reconstruction remains unestablished. This is a research
implementation, not evidence of production-scale GPU performance.

| Capability | Delivered evidence and limits |
| --- | --- |
| Helmholtz physics | Conservative second-order operator, matching sparse CSC and matrix-free actions, padded coordinate-stretched PML, optional pressure-release top, SI geometry and complex adjoints. Manufactured convergence, analytic Green functions, free-surface sign and calibrated PML reflection tests pass. |
| Differentiable inversion | Implicit first and second derivatives, source-batch gradient accumulation, bounded velocity parameters, deterministic native optimizer closures. All 15 optimizer classes in PyTorch 2.12.0 have executable CPU recipes. Selected SGD and Adam CUDA paths passed on A40; this does not certify all optimizer classes or flags on GPU. |
| Multi-resolution and multi-frequency continuation | [MultiscaleInversion](multiscale.md) executes explicit coarse-to-fine grids and low-to-high frequency bands with physical squared-slowness transfer, fixed SI acquisition and fresh native optimizer/scheduler state. CPU tests cover masks, regularizers, lifecycle failures and final-stage curvature. Two A40 Adam tests match CPU direct for PML and free-surface stages. Small examples verify orchestration, not large-scale reconstruction. |
| Curvature and uncertainty | Physical and dense-parameter exact/Gauss–Newton Hessian actions, active-cell restriction, guarded explicit diagnostics, proper-prior local covariance actions, stochastic diagonal estimates and low-rank modes. Camembert reference comparisons support these claims; uncertainty remains conditional on the stated local Gaussian/noise/prior model. |
| Outputs and restart | Bounded HDF5 writer with backpressure, optional selected arrays, accepted-state callbacks and atomic native-state/RNG checkpoints. Restart validates the actual velocity bounds and representation. Three A40 pinned-buffer/event recording tests passed; recording throughput and checkpoint portability are separate questions. |
| Dispersion and resource planning | Angular finite-difference symbol screening and immutable resource estimates. A returned `screened_only` plan is not physical validation. Multiscale execution validates every declared stage before optimization, including minimum-velocity dispersion, common domain, top boundary and nonshrinking physical PML widths. Grid/frequency schedules and PML calibration remain caller choices. |
| Iterative solver | Restarted PyTorch FGMRES and shifted Galerkin multigrid pass CPU checks, selected A40 precision/Taylor tests and one small complex128 GPU benchmark. The historical CPU sweep retained two harder convergence failures; the measured GPU workflow was slower than CPU direct. Sparse-direct CPU solves remain the reference default. |
| User workflow | Public facade, bounded JSON/CLI smoke runs, Camembert/Marmousi examples, illustrated executable README, three [executed tutorial notebooks](../notebooks/README.md), migration guides and CPU/manual-CUDA CI definitions. |

## Measured examples

The [physics report](validation.md) records approximately second-order smooth
convergence, 5 Hz homogeneous Green errors of 0.558% and 0.139% on 10 m and
5 m grids, and calibrated PML contamination below 0.1%. These fixture-specific
results do not certify arbitrary heterogeneous boundaries or frequencies.

The [full Marmousi CPU manifest](../benchmarks/fwi2d/marmousi-full-cpu.json)
uses the supplied 121 × 373 model, 25 m spacing, 800 m PML, 2 Hz, 371 sources,
371 receivers and batches of four sources. Forward modeling took 7.27 s and
the physical gradient took 22.67 s, with a measured process peak of
792,502,272 bytes (about 793 MB). The two model states required two numeric LU
factorizations. Maximum forward relative residual was 1.52e-14. Timings are
single local CPU measurements, not portable performance guarantees.

That experiment uses same-grid synthetic data. Full Marmousi PML calibration,
independent-discretization accuracy, inversion recovery and marginal benefit of
additional sources are unmeasured. At the declared minimum velocity and 1%
local phase-wavenumber tolerance, the 25 m grid's screening ceiling is about
4.31 Hz. A “highest usable frequency” has not been physically established.
Refinement is required before higher-frequency accuracy studies.

The [iterative solver results](../benchmarks/fwi2d/solver-cpu.json) preserve
both successful and failed cases and explicit waveform/gradient/HVP accuracy
criteria. FP16/BF16/integer Helmholtz quantization is outside this implementation.
Complex64 remains opt-in and requires device-specific accuracy checks.

The [multiscale CPU artifact](../benchmarks/fwi2d/results/multiscale-cpu.json)
records quick and default Marmousi crops at source revision `abbfe1f`: 5×7 to
9×13 and 9×13 to 17×25 nodes, respectively. Each runs coarse 2 Hz, fine 2 Hz and
fine joint 2/3 Hz stages with three Adam steps per stage. Each stage's final loss
is below its own initial loss. The short PML is uncalibrated, and the synthetic
data use the fine grid; these are workflow checks, not independent recovery,
high-frequency scaling or GPU performance evidence.

The separate [matched CPU/GPU run](../benchmarks/fwi2d/results/runpod-multiscale.json)
executes the quick crop on A40. All six initial/final stage-loss comparisons agree
within `6.22e-10` relative error. CPU direct took 1.03665 s; GPU FGMRES took
13.07533 s and peaked at 17,673,728 allocated CUDA bytes. These single-run timings
show no speedup. Although each stage finishes below its initial loss, the first
step of stage three increases loss and final model RMSE is 34.8429 m/s versus
33.4132 m/s for the initial constant model. Loss reduction is not recovery proof.

## Verification and hardware scope

Run the new suite with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/fwi2d -m 'not cuda' -q
```

The current CPU run completed with **463 passed, zero failures/errors/skips,
and 10 CUDA tests deselected**, as recorded in the
[multiscale CPU artifact](../benchmarks/fwi2d/results/multiscale-cpu.json).
This includes 96 multiscale CPU cases. Independent review prompted and verified
repairs for float32 frozen-bound transfer and configured scheduler ownership
and freshness. This historical CPU selection did not execute hardware tests;
their separate result is recorded below.

At the same source revision `abbfe1f`, **10 CUDA tests passed with zero failures,
errors or skips in 21.26 s** (pytest-reported) on NVIDIA A40, driver 580.178.04, Python 3.12.3,
PyTorch 2.12.0+cu130 and CUDA 13.0. The
[hardware record](../benchmarks/fwi2d/results/runpod-multiscale.json) identifies
actual `cuda:0` tensors and preserves the returned JUnit evidence. Coverage is
limited to four precision/Taylor cases including small SGD inversions, one
allocation case, three recording cases and two Adam multiscale cases.

The locked Python 3.12 / PyTorch 2.12.0 CPU environment was installed into a fresh
virtual environment, a wheel was built, and its README example ran outside the
checkout. The repository-wide legacy collection still aborts on the pre-existing
BB inversion import; see [validation](validation.md). The new suite does not
replace or conceal that result.

The three tutorials subsequently executed all **30 default code cells** on CPU
with **16 saved figures**, independent headless execution and scientific/visual
review. They demonstrate Marmousi forward solves, Adam/RMSprop/L-BFGS workflows,
and Camembert exact/Gauss–Newton curvature with local uncertainty.
Notebook 01 then executed its unchanged 11 code cells on the A40 environment,
including its bounded CUDA comparison. Notebooks 02 and 03 remain CPU-only
validation. The supplemental GPU execution has no embedded plots; the public
notebooks retain all 16 reviewed CPU figures. See the
[notebook execution record](../benchmarks/fwi2d/results/runpod-notebook-01-summary.json).

RunPod API-v2 account, billing and log access now work. A corrected reporting
harness completed the CUDA tests, CPU/GPU benchmarks, matched multiscale examples
and optional notebook 01 without changing scientific source or numerical gates.
The earlier relative-only residual-reporting failure remains in the history;
the final checker honors the configured relative-or-absolute solver criterion.
The completed test pod was deleted and its absence verified; existing global
storage was preserved. The
[attempt history](../benchmarks/fwi2d/results/runpod-attempts.json) and
[current record](../benchmarks/fwi2d/results/runpod-multiscale.json) keep these
outcomes separate. No GPU speedup is claimed. The
[bounded protocol](../benchmarks/fwi2d/runpod_protocol.md)
retains the cumulative USD 10 ceiling, at least USD 2 reserve, external deadlines,
owned-resource cleanup, and separate attributed billing.

## Implementation decisions

Work stayed in the existing isolated cloud checkout, as required by the setup
workflow. Agents implemented disjoint modules in parallel and the controller
integrated changes and commits. This avoided shared-index conflicts; interface
assumptions were checked through integration tests and independent review.
The CPU reference remains the default because the iterative scaling gate failed.
Further solver architecture work needs a measured design decision before any
claim of large-scale high-frequency performance.
