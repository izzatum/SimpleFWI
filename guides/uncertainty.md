# Local Gaussian uncertainty

`simplefwi.fwi2d.uncertainty.LocalGaussian` approximates uncertainty at a frozen
physical model `m = 1/v²`. Vectors have shape `[n_active]`, ordered by the
curvature object's active mask in array order `[z, x]`. Covariance units are
`s⁴/m⁴`; these are neither velocity nor bounded optimizer-theta uncertainties.
Recreate curvature after changing the reference model or objective settings.

```python
import torch
from simplefwi.fwi2d.curvature import Curvature
from simplefwi.fwi2d.uncertainty import LocalGaussian, CGConfig

# objective and m are an existing FWIObjective and physical slowness-squared.
curvature = Curvature(objective, m, active_mask=active_mask)
# An explicitly chosen, proper independent prior in physical m coordinates.
prior_std = 0.1 * m[active_mask]
posterior = LocalGaussian(curvature, prior_std.square().reciprocal(),
                          solve_config=CGConfig(rtol=1e-11, atol=0, maxiter=500))
variance = posterior.diagonal_estimate(probes=256, seed=19)
print(variance.mean, variance.standard_error)
```

The precision is **data-only** Gauss–Newton plus the supplied prior, exactly
once: `Re(Jᴴ Wᴴ W J) + P`. `gn_hvp` and `exact_hvp` are deliberately not used;
regularization already present in the inversion objective is not automatically
a second prior. If that regularization is your prior, supply its precision
explicitly. A valid statistical interpretation requires objective weights `W`
to whiten the declared noise. For independent complex observations the
objective `1/2 ||W residual||²` treats each real and imaginary residual component
as having the declared standard deviation. Arbitrary numerical misfit scaling
changes the inferred covariance. Metadata exposes coordinate units, active
mask and the fixed whitening weights when supplied by `Curvature`.

A finite strictly positive diagonal prior is mandatory. A pure spatial gradient
prior has a constant nullspace and is rejected. Optional smoothing can be
provided as `PriorPrecision(diagonal, smoothing=SmoothingPrecision(L, LT))`;
`L` and `LT` must be linear Euclidean adjoints and their contribution is `LT(L(v))`.
The diagonal and factor must use the same physical active-coordinate convention.
Choose boundary conditions, units, grid spacing and cell-area factors explicitly.
A positive diagonal makes this prior proper even when the smoothing factor has
a nullspace. No damping or pseudoinverse is silently introduced.

`covariance_action(v)` applies the inverse precision through diagonal
preconditioned CG. It returns a tensor and updates `last_solve` with convergence,
iteration count, true absolute/relative residuals, and a conservative solution
error bound `||residual|| / min(prior_diagonal)`. Normalized RHS arithmetic handles
tiny input vectors without treating them as zero under the default `atol=0`.
Nonconvergence and encountered nonpositive directions raise `PrecisionError`.
The API accepts a data-GN curvature object; passing an exact-Hessian action is
rejected. An arbitrary caller-provided operator's global PSD property cannot be
certified matrix-free: its mathematical symmetry/PSD contract remains the
caller's responsibility. Quadratic-action and projected-symmetry checks reject
encountered violations. Underlying Helmholtz solve errors also propagate; the
CG bound assumes accurate PSD operator actions and does not bound wave-solver
approximation error.

`diagonal_estimate(probes, seed)` streams Rademacher probes using
`z * covariance_action(z)`. `Estimate.mean` estimates **the diagonal of the
inverse**, which generally differs from reciprocal precision diagonal.
`standard_error` is the pointwise sample standard deviation divided by the
square root of probe count; at least two probes are required. It describes
Monte Carlo sampling error, not a simultaneous confidence interval, prior
uncertainty, nonlinear modeling error or solve error. Negative finite-sample
variance estimates are possible and are not clipped into false certainty.
`solve_reports` and `max_solve_error_bound` report CG error separately. Fixed seeds
reproduce estimates on the same numerical platform without changing global RNG.

`leading_modes(rank, seed, max_basis=64, max_bytes=64*1024**2)` explores a bounded
prior-metric Lanczos space, with full reorthogonalization. It solves the
generalized data problem `H_data u = lambda P u`, normalized by `uᵀ P u = 1`.
For a smoothing prior, inverse prior actions use CG; for a diagonal prior they
are exact elementwise operations. No square root of a general prior, dense
Hessian, or dense covariance is assembled. The correction is

`C_rank(v) = P⁻¹v - U diag(lambda/(1+lambda)) Uᵀv`.

`Modes` reports eigenvalues, physical vectors, absolute generalized Ritz
residual norms, explored basis dimension, whether the full parameter space was
explored, and prior solve reports. `Modes.covariance_action` also exposes its
most recent prior solve through `posterior.last_prior_solve`. This diagnostic
ranks information relative to the prior, not ordinary covariance eigenvectors.
When the basis cap truncates exploration, returned values are leading Ritz
values **within that space**, not a certified global leading spectrum. Residuals
measure the displayed modes, not a bound on all omitted information. Finite-rank
corrections retain the prior variance in omitted directions and generally
overestimate variance when they span exact leading modes. Only an exhausted
full basis with all modes sets `exact_covariance=True`; solver and roundoff
errors still apply. Workspace scales with `n_active * max_basis` and a memory
check rejects requests above `max_bytes` before building the basis.

Run the reference diagnostic from the repository root:

```bash
python examples/fwi2d/camembert_uncertainty.py --probes 8
python -m pytest tests/fwi2d/test_uncertainty.py -q
```

The example uses the pinned 48×48 Camembert domain at 5 Hz, eight surrounding
sources, 32 receivers and a **2×2 active window**, permitting a guarded 4×4
explicit covariance comparison. Its same-grid synthetic observations validate
local operator algebra, not reconstruction accuracy or calibrated geological
uncertainty. The reference is the initial homogeneous model, not an optimized
MAP estimate: this is a local GN approximation rather than a claimed exact
Laplace posterior. Positivity bounds, nonlinear/non-Gaussian behavior,
multimodality, unmodeled physics, acquisition errors and uncertain noise/prior
hyperparameters are outside this approximation.
