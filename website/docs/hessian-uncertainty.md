---
title: FWI Hessian and Gauss–Newton uncertainty in PyTorch
description: Compute matrix-free exact and Gauss–Newton Hessian-vector products for 2D FWI, and explore proper-prior local Gaussian uncertainty on a Camembert model.
---

# FWI Hessian and Gauss–Newton uncertainty

**SimpleFWI computes exact and Gauss–Newton Hessian-vector products without
assembling a dense Hessian.** These actions support curvature diagnostics and
local uncertainty calculations around a fixed physical model. The resulting
uncertainty depends on declared noise scaling, parameter coordinates, a proper
prior and the local approximation.

## Exact Hessian versus Gauss–Newton Hessian

For a weighted least-squares data objective, Gauss–Newton curvature uses the
Jacobian action and its adjoint, `Re(Jᴴ Wᴴ W J)`. The exact Hessian additionally
includes residual-dependent second derivatives. It can be indefinite away from
a suitable local minimum. They answer different questions and are not
interchangeable positive precision operators.

`Curvature` freezes a physical reference `m = 1 / v²`. Its `exact_hvp` and
`gn_hvp` accept real vectors over an explicit active mask in `[z, x]` order.
`TransformedCurvature` labels bounded parameter-coordinate actions separately;
exact transformed curvature includes the nonlinear map's second derivative.
Recreate curvature after changing the reference model or objective.

Use the [curvature implementation and public API](https://github.com/izzatum/SimpleFWI/blob/main/simplefwi/fwi2d/curvature.py)
and [uncertainty guide](https://github.com/izzatum/SimpleFWI/blob/main/guides/uncertainty.md)
for coordinate conventions and complete examples.

## What does local Gaussian uncertainty require?

`LocalGaussian` applies the inverse of **data-only Gauss–Newton curvature plus
an explicitly supplied proper prior precision**, counted once. Objective
weights must whiten the declared observational noise for a statistical
interpretation. Numerical rescaling of the misfit changes the covariance.

- A finite strictly positive diagonal prior is mandatory; optional smoothing
  may be added with a declared adjoint. No damping or pseudoinverse is silently
  inserted.
- Covariance actions use preconditioned conjugate gradients with convergence
  checks and true residual reports. Nonconvergence and encountered nonpositive
  directions raise errors.
- Streaming Rademacher probes estimate the diagonal of the inverse precision,
  which generally differs from the reciprocal precision diagonal. Sampling
  standard errors are reported separately from solver errors.
- Bounded prior-metric Lanczos exploration returns information modes and Ritz
  residuals. A truncated basis does not certify the full global leading spectrum.

Physical squared-slowness covariance has units **s⁴/m⁴**. It is not velocity
variance or unconstrained optimizer-parameter variance. Local Gaussian results
do not capture all nonlinear effects, multimodality, unmodeled physics or
uncertain noise and prior hyperparameters.

## Verify the algebra on Camembert

From an [installed checkout](index.md#install-the-validated-cpu-environment):

```bash
python examples/fwi2d/camembert_uncertainty.py --probes 8
```

The reference example uses a **48 × 48** Camembert domain at **5 Hz**, eight
surrounding sources, 32 receivers and a **2 × 2 active parameter window**. Four
active parameters permit a guarded explicit 4 × 4 covariance comparison while
retaining the wave physics on the full grid. Dense assembly is a small-test
diagnostic, not the production strategy for a large model.

This example uses same-grid synthetic observations and an initial homogeneous
reference, not an optimized MAP estimate. It verifies local operator algebra;
it does not demonstrate calibrated geological uncertainty or reconstruction.
Read the
[Camembert Hessian and uncertainty notebook](https://github.com/izzatum/SimpleFWI/blob/main/notebooks/03_camembert_hessian_uncertainty.ipynb)
for figures and exact/Gauss–Newton comparisons, then the
[validation record](validation.md) for evidence boundaries.
