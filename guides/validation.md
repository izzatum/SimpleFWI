# Independent acoustic physics validation

The reference convention is SI units, `[z, x]` arrays, `exp(-i omega t)`,
and `A = Delta + omega**2/v**2`. A unit positive point source is scattered
with `1/(dz*dx)` normalization. Its outgoing two-dimensional field is
`-i H0^(1)(omega*r/v)/4`; using the opposite sign changes the source.
All results below use CPU float64 models and complex128 fields.

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

## Current CPU evidence, unrun CUDA checks and legacy failures

The [CPU solver manifest](../benchmarks/fwi2d/solver-cpu.json) records direct and
FGMRES comparisons, including failed-convergence cases. Local iterative timings
were worse than direct solves; these measurements establish no GPU speedup.
The [full-domain Marmousi CPU manifest](../benchmarks/fwi2d/marmousi-full-cpu.json)
records finite forward data and physical gradients at 2 Hz on the supplied
25 m grid, including 371 nested source positions. It does not establish
independent physical accuracy, reconstruction quality, a highest useful
frequency, or the information gained from additional sources.

Actual CUDA execution remains unrun. Two subsequent RunPod attempts reached
provisioned hardware but failed remote execution access: direct SSH was
unreachable and the explicitly exposed, authenticated Jupyter proxy returned
Cloudflare error 1010. Both pods were terminated/deleted with provider-confirmed
absence; no persistent storage remains. The
[attempt manifest](../benchmarks/fwi2d/results/runpod-attempts.json) records this
separately from the successful CPU numerical work.
The eight CUDA-required checks cover four device/precision cases, one allocation
scaling case, and three pinned-buffer/event recording cases. CPU runs report
these as skips; they are not passing GPU validation. The
[RunPod protocol](../benchmarks/fwi2d/runpod_protocol.md) describes the bounded
hardware validation procedure and records the remote-execution blocker.

Use `python -m pytest tests/fwi2d -q` for the new implementation's targeted suite.
The legacy full-repository pytest run still aborts during collection: the
unbounded BB script performs work on import and encounters a singular
factorization, while stale scripts import the removed `simplefwi.core_v2` module.
Preserved legacy entry points do not make those historical tests pass. No
whole-repository green claim is made; the measurements above retain their stated
fixture, device and numerical scope.
