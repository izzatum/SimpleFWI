# Spatial screening and fixed optimization stages

`assess_dispersion` evaluates the homogeneous symbol of the conservative
second-order stencil. It uses the **smallest allowed velocity**, rather than
only the velocity of the current iterate, and the largest requested frequency.
Distances are metres, velocities m/s and frequencies Hz. Zero and nonfinite
frequencies are errors. A rectangular grid has separate `dz` and `dx`.

For angle θ measured from x towards z and numerical wavenumber κ, the symbol is

```
S(κ,θ) = 4 sin²(κ dz sinθ/2)/dz² + 4 sin²(κ dx cosθ/2)/dx².
```

The assessment solves `S(κ,θ) = (2π f/v_min)²` on the monotone first Brillouin
branch at 721 angles in `[0,π/2]`; symmetry covers other quadrants. The reported
relative phase-wavenumber error is `κ/(2πf/v_min) - 1`. If no root exists on that
branch, the error is infinite and the configuration fails. This quantity is a
wavenumber error, not a group-velocity or waveform error.

The default target is at most 1%. A conservative frequency ceiling for all
angles uses `h = max(dz,dx)`. With `q = π f h/v_min`, solve
`asin(q)/q - 1 = tolerance`; then `f_ceiling = v_min q/(πh)` and the required
points per wavelength are `p = π/q`. This bound includes the axial direction
along the largest grid spacing and does not depend on accidentally missing a
worst angle. For the 1% target, approximately 13.00 points per wavelength are
needed. A 10 m grid at a 1500 m/s lower bound passes 10 Hz but fails 15 Hz;
5 m spacing passes 15 Hz. This is a local homogeneous screening result, not
a global guarantee for heterogeneous data, interfaces or point sources.

```python
from simplefwi.fwi2d.types import BoundaryConfig, Grid2D
from simplefwi.fwi2d.dispersion import assess_dispersion
from simplefwi.fwi2d.planning import StageRequest, plan_stage

grid = Grid2D((48, 48), (10., 10.))
report = assess_dispersion(grid, (1500., 2500.), (5., 15.))
assert not report.passed
request = StageRequest(
    grids=(grid, Grid2D((95, 95), (5., 5.))),
    velocity_bounds=(1500., 2500.), frequencies_hz=(5., 15.),
    source_count=8, receiver_count=32, source_batch_sizes=(1, 2, 4),
    boundary=BoundaryConfig(pml_cells=20),
)
plan = plan_stage(request, memory_budget_bytes=512 * 1024**2)
assert plan.grid.spacing_m == (5., 5.)
assert plan.validation_status == 'screened_only'
```

`StageRequest` accepts at most 64 explicit grid candidates and 64 batch sizes.
Candidates must share node extents and origins. Planning searches by increasing
physical cell count, then decreasing source batch size. Frequencies run
sequentially. It keeps requested frequencies, tolerance, complex128 precision
and PML settings fixed. It raises `PlanningError` if no candidate passes spatial
screening or if every passing candidate exceeds the supplied memory budget.
Only the second-order stencil is available; higher-order alternatives require
independent operator, physics and derivative validation before registration.

The memory estimate counts physical real model/gradient arrays, PML padding,
complex coefficient/work arrays, four wavefields per batch, both flexible
Krylov bases (`V` and `Z`) and Hessenberg work, L-BFGS history storage, and
observed/predicted/residual data. It adds a 25% temporary/allocator reserve.
A free surface omits top PML padding. The estimate is for matrix-free FGMRES;
it does not predict sparse direct factorization fill or arbitrary external
preconditioner/cache allocations. Reserve known additional memory through
`extra_workspace_bytes` and measure actual peaks before a large run. A plan's
estimate is not a hard runtime memory guarantee or evidence of solver scaling.

A plan is a frozen value containing a frozen grid, boundary and report.
Construct it before optimizer steps and preserve grid, stencil, PML, source
set, weights and frequencies throughout all closure evaluations in a step.
Model parameters can change without changing the plan. At a reviewed stage
boundary, map physical models by their SI coordinates, preserve velocity bounds,
rebuild acquisition/source interpolation, invalidate numerical caches and reset
incompatible optimizer histories after changing grids or objectives. The planner
does not perform model interpolation or optimizer mutation.

PML reflection and true linear residuals are independent acceptance gates.
Every returned plan records `pml_validation='required'` and
`linear_solve_validation='required'`; `screened_only` must not be treated as
physical validation. Compare data with a refined-domain reference using the
same physical model and acquisition, document near-source exclusions, and sweep
PML thickness separately from spacing. The PML reflected-amplitude target is
`1e-3` on calibrated fixtures across frequencies and 0°, 30° and 60° incidence.
Reject failed solves before accepting gradients or optimizer updates. There is
no time-stepping CFL condition in this frequency-domain calculation.
