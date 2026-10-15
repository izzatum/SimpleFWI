# SI acoustic architecture

`simplefwi.fwi2d` exposes the new PyTorch workflow alongside preserved legacy
exports. `Grid2D` defines physical **nodes**, with shape `(nz, nx)`, spacing
`(dz, dx)` in metres, and explicit first-node origin `(z0, x0)`. A physical model
is `[z,x]` velocity in m/s; physical optimization coordinates are
`m = 1/v²` in s²/m². `Acquisition` supplies `(z,x)` coordinates in metres and
one complex scalar amplitude per source. Bilinear receiver interpolation and
its Euclidean transpose scatter use identical weights.

The constant-density operator discretizes
`(∇² + ω² m)p = q`, using Fourier convention `exp(-i ω t)` and positive
`ω = 2π f`. `SamplingOperator.source_rhs` divides scattered amplitudes by
`dz*dx`, so integrated source strength is grid-area normalized. Unit amplitudes
are normalized numerical source units; pressure and sampled data are labelled
`source_unit`, **not calibrated pascals**. Compare datasets only when source
normalization and amplitudes match. The weighted data loss is
`0.5 * ||W * (predicted-observed)||²`, with an optional physical regularizer.

`BoundaryConfig` pads physical nodes with cubic stretched-coordinate PML on
all four sides. `top='free_surface'` removes top padding and imposes pressure
zero at the top physical row. Other outer pressure boundary degrees of freedom
are eliminated. Source/receiver points exactly on an eliminated boundary are
rejected. The optional free surface is flat; topography is outside this model.
Velocity is extended into PML consistently so physical derivatives include that
extension's transpose. PML thickness and damping require accuracy checks.

`AcousticFWI.simulate` returns detached complex data in
`[frequency,source,receiver]` order. Batches of wavefields have
`[source_batch,z_padded,x_padded]` axes when unpacked; compact solve vectors
contain only unconstrained degrees of freedom. `objective` returns
`FWIObjective`; implicit differentiation solves the adjoint rather than
backpropagating through the factorization or Krylov iterations. A failed solve
raises before its gradient can be accepted.

The direct solver assembles a sparse CPU reference system and uses bounded
factor caches; matrix-free here describes iterative operator/Hessian actions,
not a promise that direct factorization allocates no matrix. FGMRES uses a
shifted-Helmholtz multigrid preconditioner and true unpreconditioned residual
acceptance. It has CPU tests but can fail at higher frequencies; actual CUDA
execution is unrun. No implicit device or precision fallback is supported.

`VelocityParameters` maps bounded native parameters to physical velocity.
`InversionRunner` accumulates streamed physical gradients and applies one
pullback to the parameterization per closure. Native `torch.optim` controls the
update. See the [optimizer matrix](optimizers.md), including SparseAdam and
shape requirements. Each closure keeps acquisition, weights and discretization
fixed. Native weight decay acts in parameter coordinates, independently of the
physical regularizer.

`Curvature(objective, model.slowness_squared())` freezes its physical reference.
`exact_hvp(vector)` and `gn_hvp(vector)` accept real `[n_active]` vectors and
return actions without assembling a dense Hessian. `TransformedCurvature`
labels theta-coordinate actions separately and includes the nonlinear mapping's
second derivative for exact curvature. Recreate either object after changing
the reference. The [uncertainty guide](uncertainty.md) covers coordinates,
positive precision requirements, probes, and interpretation limits.

`AcceptedStateRecorder` binds explicitly selected HDF5 outputs and atomic
checkpoints to accepted outer steps. It can recompute the accepted-state gradient;
a closure's last gradient may instead belong to a line-search trial. Simulation
wavefields use separate selected archives. CPU recording uses bounded buffers;
CUDA recording uses bounded pinned buffers and transfer events, but actual-device
checks remain unrun. See [output/restart](outputs.md) for
metadata, identities, writer failure behavior, and storage limits. Stored data,
optimizer state, and sparse factors remain separate memory costs.
