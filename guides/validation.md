# Independent acoustic physics validation

The reference convention is SI units, `[z, x]` arrays, `exp(-i omega t)`,
and `A = Delta + omega**2/v**2`. A unit positive point source is scattered
with `1/(dz*dx)` normalization. Its outgoing two-dimensional field is
`-i H0^(1)(omega*r/v)/4`; using the opposite sign changes the source.
The independent physics results below use CPU float64 models and complex128
fields. The broader implementation suite also includes explicit float32
parameter and multiscale transfer regressions.

Run the focused scientific gate from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/fwi2d/test_physics.py tests/fwi2d/test_pml_reflection.py \
  tests/fwi2d/test_camembert.py -q -s
```

The historical focused run measured **27 passed in 56.32 s**, with no skips or
warnings. This is the count for that recorded physics run, not the current full
`tests/fwi2d` suite.
The thread limits avoid oversubscription during concurrent development.
`-s` prints the scalar observations; no reference wavefields are stored.

## Smooth solutions, source amplitude and free surface

The manufactured problem has independently differentiated sine modes and a
smooth variable velocity on a 400 by 600 m rectangle with origin (-70,120) m.
Its `(dz,dx)` refinements are (20,30), (10,15), and (5,7.5) m. Homogeneous
Dirichlet values are exact. The RHS uses the analytic Laplacian, not the
implemented discrete operator applied to a chosen field.

| Check | Measured result | Acceptance |
|---|---:|---:|
| Manufactured relative L2 errors | 6.25452e-3, 1.55982e-3, 3.89718e-4 | Decrease under refinement |
| Observed orders | 2.00352, 2.00088 | Each between 1.8 and 2.2 |
| 5 Hz homogeneous Green error, 10 m grid | 5.57866e-3 | Below 1% |
| 5 Hz homogeneous Green error, 5 m grid | 1.38999e-3 | Below 1%; improvement 3.5–4.5 times |
| Pressure-release image field relative error | 1.44644e-3 | Below 1% |
| Fitted pressure image coefficient | -1.001492 - 0.002737i | Distance from -1 below 0.02 |

Green-function probes are fixed in metres and exclude every point within
80 m of the source. Absolute complex values are compared without fitting a
phase or amplitude. The image test subtracts the reflected Green function
at the mirror source above `z=0`; it explicitly rejects the positive-image
(rigid boundary) field. These tests also check true solver residuals.

## PML reflection calibration

Cubic damping uses `sigma_max=60 /s`, `alpha=0`, with stretches evaluated on
faces and centres by the production operator. Plane incidence tests extract
an interior tangential Fourier section from the assembled two-dimensional
operator. The tangential direction is infinite in this experiment, so the
incident numerical wavevector is exactly at 0, 30 or 60 degrees to the
x-boundary normal. The numerical interior dispersion relation fixes that
wavevector; spatial phase error is assessed separately.

Each reflection measurement compares a 600 m physical strip against a
1200 m strip with a 1600 m PML. The expanded reference is independently
checked against the exact outgoing discrete strip Green function to within
1e-5. Reflection is measured both as maximum complex field perturbation
relative to incident amplitude and as the incoming-wave coefficient in a
two-wave fit. Both must be <=1e-3. Probe positions are 60–150 m from the
source, in 10 m steps.

Thicknesses of 800 and 1000 m are swept at **each** of 10 and 5 m spacing,
with physical thickness held fixed during refinement. Frequencies are
2.5, 5, 10 and 15 Hz. The 2.5 Hz case is the explicit low-frequency test.
Increasing thickness must improve each fixed-grid result. This distinguishes
PML termination error from discretization error.

The maximum measured plane contamination over both spacings and all four
frequencies was:

| Incidence angle | 800 m layer | 1000 m layer |
|---|---:|---:|
| 0 degrees | 2.22298e-7 | 7.49270e-9 |
| 30 degrees | 1.81756e-6 | 5.58246e-8 |
| 60 degrees | 6.61167e-4 | 8.88898e-5 |

An exploratory 600 m layer failed the 60-degree gate, with approximately
4.9e-3 contamination. It was not accepted. The pinned Camembert boundary
therefore uses 800 m (80 cells at 10 m), not that cheaper failed setting.
The layer is deliberately conservative; no minimal-cost PML claim is made.

The separate full two-dimensional test samples rays approaching **all four
corners** at coordinates `(+-r,+-r)`, r=120,180,240 m. It compares a 600 m
square against a 1000 m square with an 800 m PML, on the same 10 m grid.
The measured normalized perturbation must remain <=1e-3 for both 400 and
600 m layer thicknesses at 2.5, 5, 10 and 15 Hz. This checks simultaneous
x/z stretching; it is not presented as a plane-incidence experiment.

| Corner-test frequency | 400 m layer | 600 m layer |
|---|---:|---:|
| 2.5 Hz | 5.73223e-4 | 8.99781e-6 |
| 5 Hz | 4.90595e-4 | 7.50199e-6 |
| 10 Hz | 5.43888e-4 | 7.60638e-6 |
| 15 Hz | 4.30143e-4 | 5.59357e-6 |

Finite PML is not exactly absorbing. These are homogeneous fixture results,
not guarantees for arbitrary heterogeneous edge velocities, grazing angles,
frequencies, topography or complex-frequency-shifted PML. A 15 Hz PML test
on the 10 m grid isolates the boundary behavior; it does not override that
grid's failed continuum dispersion screen.

## Camembert and independent data

The exact fixture is [camembert.json](../tests/fwi2d/fixtures/camembert.json).
It fixes 48 by 48 physical nodes at 10 m spacing, origin (0,0), a circle
centred at (235,235) m of radius 60 m, 1500 m/s background, and 2000 m/s
inclusion. The initial model is homogeneous 1500 m/s. Half-open active
windows `[16:32,16:32]` and `[12:36,12:36]` provide 16 by 16 and 24 by 24
parameter windows.

Eight surrounding sources and 32 distinct receivers have exact coordinate
arrays on the perimeter of `[40,430] m` squared. They do not coincide.
The separate surface geometry has sources at z=40 m and receivers at z=60 m;
selecting it does not implicitly change the top boundary condition.
Source amplitudes are one and observations have `[frequency,source,receiver]`
axes. Candidate frequencies are 5, 10 and 15 Hz. At the 1500 m/s minimum
velocity, 5 and 10 Hz pass the <=1% local phase screen on 10 m spacing;
15 Hz is explicitly **algebraic-only on the 10 m grid**. The 5 m reference
passes that local screen through 15 Hz. A local screen alone does not imply
1% heterogeneous waveform accuracy.

[camembert.py](../examples/fwi2d/camembert.py) provides `make_camembert`,
`make_refined_camembert`, and `simulate_data`. The independent fixture uses
5 m spacing, origin (-100,-100) m, 135 by 135 physical nodes, and a 160-cell
PML preserving 800 m thickness. Circle centre, radius, velocities and
acquisition remain unchanged in metres. The heterogeneous interface is
sampled geometrically; it must not be confused with smooth second-order
manufactured convergence.

For a configured pressure-release top, the reference keeps the original
physical z-origin and adds vertical margin only at the bottom. Horizontal
margins still extend both sides. Source and receiver depths below the free
surface therefore remain unchanged, including with a nonzero grid origin.

At 5 Hz, with all eight sources and 32 receivers, the coarse true-model
data differ from the larger/finer reference by **1.65160%** in relative
complex L2 norm. The initial-model data differ from that reference by
**11.31138%**. These are measured discrepancies, not an asserted continuum
error bound. There is no claim of converged heterogeneous reference accuracy
from two grids alone. Same-grid data are appropriate for algebraic derivative
checks; independent refined data are required when assessing reconstruction
accuracy. The present tests establish reference generation, not inversion
quality or success at every candidate frequency.

## CPU evidence, selected CUDA checks and limits

The [multiscale CPU artifact](../benchmarks/fwi2d/results/multiscale-cpu.json)
records the current `tests/fwi2d` gate at revision `abbfe1f`: **463 passed, zero
failures/errors/skips, and 10 CUDA tests deselected**. Reproduce that CPU selection
with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/fwi2d -m 'not cuda' -q
```

Its 96 multiscale CPU cases cover node-aligned physical-m refinement, exact
frequency/weight selection, one physical prior, fresh optimizer/scheduler state,
explicit masks, free-surface/PML geometry, failure cleanup and final-stage
curvature. Real PDE Adam/LBFGS runs match independently constructed stage runners.
The artifact also records three-stage quick/default Marmousi crop examples with
within-stage loss reduction. Their small grids and uncalibrated short PML do not
establish independent continuum accuracy, full-model recovery, high-frequency
scaling or GPU performance. See the [continuation guide](multiscale.md) for the
physical transfer and schedule contracts.

The [CPU solver manifest](../benchmarks/fwi2d/solver-cpu.json) records direct and
FGMRES comparisons, including failed-convergence cases. Local iterative timings
were worse than direct solves; these measurements establish no GPU speedup.
The [full-domain Marmousi CPU manifest](../benchmarks/fwi2d/marmousi-full-cpu.json)
records finite forward data and physical gradients at 2 Hz on the supplied
25 m grid, including 371 nested source positions. It does not establish
independent physical accuracy, reconstruction quality, a highest useful
frequency, or the information gained from additional sources.

The separate [hardware record](../benchmarks/fwi2d/results/runpod-multiscale.json)
reports **10 CUDA tests passed, zero failures/errors/skips, in 21.26 s**
(pytest-reported) at source
revision `abbfe1f`. The actual device was NVIDIA A40 (`cuda:0`), driver 580.178.04,
with Python 3.12.3, PyTorch 2.12.0+cu130 and CUDA 13.0. The selected cases cover:

- Four precision/Taylor cases, including physical-m waveform/gradient parity,
  resident multigrid data, and small SGD inversions in float32/complex64 and
  float64/complex128. The waveform/physical-gradient relative gates are
  `1e-3`/`1e-2` for complex64 and `1e-7`/`1e-7` for complex128.
- One bounded allocation-scaling case and three pinned-buffer/event recording
  cases, including an injected disk-failure cleanup.
- Two float64/complex128 Adam multiscale comparisons with CPU direct, one with
  PML and one with a free surface. Resident waveform/physical-m gradient parity
  must meet `1e-7` relative error, with separate stage-loss and model checks.

Select these tests with `python -m pytest tests/fwi2d -m cuda -q`. Their actual
execution is separate from the historical 463-pass CPU selection, which
deselected all ten. The result does not certify all 15 native optimizers on CUDA,
large surveys, high frequencies or GPU speedup.

The [GPU solver benchmark](../benchmarks/fwi2d/results/runpod-solver-cuda.json)
also passed against CPU direct on its small complex128 algebraic fixture:

| Quantity | Measured relative value |
| --- | ---: |
| Primal residual | 3.2781126e-12 |
| Waveform error | 4.5576127e-12 |
| Gradient error in log-m coordinates | 8.0647427e-12 |
| Exact HVP error in log-m coordinates | 2.6006139e-10 |

The benchmark gradient and exact-Hessian actions use
`theta = log(m/m_reference)` coordinates. Physical-m parity is established by
the separate tests above. All eight GPU benchmark solves satisfy their
configured **relative OR absolute** residual criterion. For tiny derivative
right-hand sides, the absolute branch can pass despite a larger relative ratio.
An earlier run stopped because its reporting harness ignored that branch; the
corrected checker changed no scientific source or numerical tolerances.

The matched [CPU](../benchmarks/fwi2d/results/runpod-multiscale-cpu.json) and
[GPU](../benchmarks/fwi2d/results/runpod-multiscale-cuda.json) Marmousi crops agree
within `6.22e-10` relative error across all six initial/final stage losses. Each
stage's final loss is lower, but stage three's first Adam step raises loss.
Final velocity RMSE is **34.8429 m/s**, worse than the initial constant truth-mean
model's **33.4132 m/s**. The data-loss reduction therefore does not demonstrate
improved recovery. CPU direct took **1.03665 s**, versus **13.07533 s** for GPU
FGMRES with **17,673,728 peak allocated CUDA bytes**. These single small-crop,
short-PML measurements show no speedup and do not establish large-scale memory
behavior, calibrated boundaries or high-frequency accuracy.

Notebook 01's unchanged 11 code cells also executed on A40. Its bounded CUDA
crop passed the `<1e-7` waveform/data agreement assertions; formatted table values
of `0.0` are rounding, not exact-zero errors. The supplemental GPU run contains
no embedded plots, so the public notebooks retain their reviewed CPU figures.
Notebooks 02 and 03 remain CPU-only execution evidence. See the
[notebook summary](../benchmarks/fwi2d/results/runpod-notebook-01-summary.json).

The completed test pod was deleted and its absence verified; existing global
storage was preserved. Earlier
[access failures](../benchmarks/fwi2d/results/runpod-attempts.json) remain historical
outcomes. The [RunPod protocol](../benchmarks/fwi2d/runpod_protocol.md) records the
bounded hardware procedure and cleanup; reported cost estimates remain distinct
from final provider itemization.

Use `python -m pytest tests/fwi2d -q` for the entire new implementation suite,
including hardware-required tests when CUDA is available.
The legacy full-repository pytest run still aborts during collection: the
unbounded BB script performs work on import and encounters a singular
factorization, while stale scripts import the removed `simplefwi.core_v2` module.
Preserved legacy entry points do not make those historical tests pass. No
whole-repository green claim is made; the measurements above retain their stated
fixture, device and numerical scope.
