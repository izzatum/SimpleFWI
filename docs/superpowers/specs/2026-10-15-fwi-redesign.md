# SimpleFWI 2D PyTorch redesign specification

Date: 2026-10-15. Status: consolidated design for review; implementation has not started.

## Purpose and agreed scope

Rebuild SimpleFWI as an accurate, high-performance, memory-efficient 2D constant-density acoustic frequency-domain FWI library with a PyTorch user interface. Validate the mathematics on a small Camembert model before scaling to Marmousi. Provide PML, an optional pressure-release free surface, native PyTorch optimizer integration, operator-based exact and Gauss–Newton Hessians, local uncertainty tools, and configurable recording of wavefields, model iterates, and gradients.

The user prioritizes correctness, then measured performance at equal accuracy. They request the highest useful Marmousi frequency and as many useful sources as practical. These are constrained searches with explicit accuracy and resource limits, not an unlimited workload or a property of the model file alone. Current scope excludes 3D, elasticity, anisotropy, attenuation, variable density, learned priors, distributed execution, and arbitrary neural model parameterizations. Interfaces must not preclude later extensions.

The implementation budget and duration are distinct from the RunPod budget. Total RunPod expenditure for this task must not exceed USD 10, including compute, storage and other billed resources. No GPU resources were started during planning.

## Current evidence and migration baseline

Inspected checkout: `izzatum/SimpleFWI`, HEAD `db5c9af426aae90879fc873c50edcaf8d1f2cdf0`.

- Current propagation uses NumPy/SciPy sparse factors and PyLops, not a working PyTorch GPU backend.
- The current absorbing edge treatment is not PML. Forward and Jacobian construction repeat factorizations and retain all source/frequency wavefields.
- The regularization value lacks the one-half factor consistent with its implemented gradient; its derivative acts along a flattened model instead of separate spatial axes.
- An existing unbounded BB example makes five slowness-squared cells negative in its first step (minimum approximately -0.03445), then fails factorization.
- Three legacy validation scripts import missing `simplefwi.core_v2`. A selected test group contains 52 skipped placeholders. These are not a correctness baseline.
- Forward/API checks, finite gradients, a complex Jacobian adjoint check, and notebook execution passed during onboarding. These do not establish correct physics or a correct complete objective.
- `data/marmousi.mat`: 121 by 373, 25 m spacing, x origin -200 m, velocities approximately 1466–5928 in apparent m/s. `data/marmousi_50.mat`: 61 by 220, 50 m spacing, values 1.5–4.7 in apparent km/s. The loaders must require/document unit conversion rather than infer units from magnitude.

Create the new implementation in `simplefwi/fwi2d/`, leaving legacy APIs intact during development. Gate any later public-API migration on validated examples and an explicit compatibility document. The old numerical results are regression observations, not scientific ground truth.

## Global implementation constraints

- Python 3.12 is the initial supported and tested runtime.
- PyTorch is the public tensor, parameter, optimizer, and autodiff framework; real model parameters use float64 by default and wavefields use complex128 by default.
- Pin an exact released PyTorch version and compatible dependencies when implementation begins; commit the resolved environment and test the optimizer inventory for that version. Do not depend on nightly releases or promise unknown future APIs.
- Use NumPy/SciPy for the independent CPU sparse-direct reference and small explicit validation matrices. GPU production operators use PyTorch; a specialized solver dependency requires an isolated adapter, license review, and measured benefit before adoption.
- Use SI units internally, Fourier convention exp(-i omega t), array order [z, x], and explicit coordinate origins.
- Observed/predicted data axes are [frequency, source, receiver]; batched wavefields are [source_batch, z_padded, x_padded]. Axes and units are saved with outputs.
- Use the existing checkout. Cloud tasks are already isolated; do not create a Git worktree unless explicitly requested.
- Planning changes documentation only. Implementation, dependency installation, and paid GPU experiments are subsequent work.
- Never disable assertions, silently suppress failed solves, or count skipped/unexecuted tests as passing.
- No zero-frequency Helmholtz solves, silent coordinate clipping, silent device fallback, or silent precision downgrade.

## Mathematical and acquisition contract

Let m = 1/v^2 in s^2/m^2. For each frequency, solve A(m, omega)u = q and sample d = Pu. The discrete RHS includes the selected source amplitude and cell-area normalization. Source and receiver interpolation use identical, tested weights and their exact transpose/adjoint scatter operations. Reject points outside the physical domain and pressure-release nodes. Support rectangular spacing and nonzero origins.

Use Euclidean inner products on discrete coefficient arrays; cell-area and quadrature factors belong explicitly in source normalization and regularization. The data objective is one-half ||W(d - d_obs)||^2, with W a fixed diagonal complex/real weighting tensor or scalar for the initial release. Noise whitening is represented by W. General correlated noise operators are a later extension.

The physical-model gradient is obtained from the exact discrete derivatives:

    A^H lambda = P^H W^H W(Pu - d_obs)
    g_j = -Re[lambda^H (dA/dm_j)u] + (g_R)_j.

Sum frequency/source contributions consistently. Spatial regularization is R = beta/2 ||L(m - m_prior)||^2, using physical grid spacing, cell-area weights, and separate x/z derivatives with documented boundaries. Its gradient is beta L^T L(m-m_prior), and its Hessian is beta L^T L. Do not introduce an undocumented normalized-by-data-count objective.

## PML and free surface

Pad the physical domain with computational PML cells; stretching operates in that padding. Default boundaries use PML on all four sides. The user may replace the top PML with a pressure-release free surface (p=0); the other sides retain PML. Eliminate constrained pressure degrees of freedom consistently. The initial free surface is flat; topography is outside scope.

For each direction j:

    s_j = kappa_j + sigma_j / (alpha_j - i omega).

Use kappa=1 and alpha=0 for the first validated PML. Expose a nonnegative complex-frequency shift alpha as a later validated option, not a requirement for the initial release. Sigma is zero in the physical domain and rises as a cubic profile through PML. At corners, apply both stretches. The conservative stretched equation is:

    d_x[(s_z/s_x)d_x u] + d_z[(s_x/s_z)d_z u]
        + omega^2 m s_x s_z u = q_transformed.

Place derivative coefficients on faces and mass coefficients at centres. Start with conservative second-order finite differences and independently verify sparse assembly against matrix-free action. High-order/optimized stencils are a separate gate; do not substitute naive high-order boundary closures.

Extend velocity from physical boundary cells into PML by nearest-boundary extension. Express this as an explicit map and include its derivative in model gradients and Hessians. Freeze PML profiles based on declared velocity bounds and frequency-stage settings, not the current differentiable velocity values. Include extension dependencies when a physical boundary cell is active. Only physical cells are inversion parameters.

Terminate the outside of PML with homogeneous Dirichlet conditions. Calibrate thickness/damping against a larger-domain reference. Record requested and measured reflection. A failed target produces an actionable error or explicit user-approved relaxation; no claimed exact absorption at finite thickness.

## Solvers, resources and precision

Support both representations under one operator contract:

1. CPU sparse-direct reference: reuse a numeric factorization across RHS batches at the same model/frequency, and use conjugate-transpose solve support where available. Reuse symbolic structure only when the backend supports it.
2. PyTorch matrix-free production: restarted right-preconditioned FGMRES, initially with a shifted-Helmholtz geometric multigrid preconditioner on supported structured grids. Derive compatible coarse operators and boundary handling. Plain/Jacobi-preconditioned Krylov is a diagnostic baseline, not an assumed scalable solver.
3. If the production preconditioner fails accuracy/resource gates, retain the validated direct path and report the unresolved scaling limit. Sweeping or domain-decomposition methods require a separate measured design decision, not an unbounded implementation expansion.

Ordinary CG is not the Helmholtz solver. Solver results include true unpreconditioned relative residual, absolute residual, iterations, convergence reason, elapsed time, and memory estimates. For zero RHS, use the absolute-residual rule rather than dividing by zero. Failed convergence raises a structured exception before a gradient or optimizer update is accepted.

Cache identity includes model evaluation identity, frequency, grid/origin, PML/free-surface settings, precision, device, and solver configuration. In-place optimizer updates and closure trial points invalidate numeric state. Never use tensor data pointers alone as model identity. Bound cache memory and keep no unbounded autograd graphs.

Batch sources and process frequencies sequentially by default. Use per-batch backward accumulation for first-order optimization; retaining a list of all differentiable batch losses would defeat the memory bound. Higher-order differentiation recomputes needed state or uses an explicitly budgeted checkpoint policy.

Complex128 is the reference. Complex64 is opt-in after waveform, gradient, Taylor and inversion comparisons. Residual checks/reductions may use higher precision; iterative refinement is enabled only when measured to converge. FP16, BF16 and integer quantization are not initial Helmholtz representations. Do not assume a consumer GPU has fast FP64 or that GPU is faster for the supplied small 2D grids.

## PyTorch autograd and optimizer compatibility

Expose a real scalar differentiable objective and ordinary trainable PyTorch parameters. Use implicit differentiation of linear solves rather than backpropagating through all FGMRES iterations. The solve backward uses adjoint solves and differentiable stencil products. Support second derivatives through differentiable backward/recomputation, verified with `gradcheck` and `gradgradcheck`; a detached first backward is insufficient. Explicit Hessian-vector interfaces remain the scalable preferred path.

All native optimizer algorithms in the pinned `torch.optim` release must have a documented, tested valid integration route. This does not mean every optimizer accepts the same parameter representation, every optional implementation flag, or guarantees convergence on FWI.

- Default dense parameter: a real two-dimensional `torch.nn.Parameter` theta, mapped to bounded physical velocity by v = v_min + (v_max-v_min) sigmoid(theta). Validate interior initial values, expose physical gradients separately, and report saturation. The regularizer/prior is defined in physical m, not implicitly in theta.
- SparseAdam route: explicitly opt into a sparse-gradient `nn.Embedding` parameter container for active model cells, gather its entries, then map/scatter into the physical model. This changes gradient representation, not the physical objective; all selected cells may have nonzero gradients, so no sparse-computation speedup is promised. Default dense gradients must never be passed to SparseAdam.
- Shape-constrained optimizers, if present in the pinned release, receive valid parameter shapes or a documented alternative adapter. Build an explicit recipe registry against the discovered stable optimizer classes; a newly exported class without a recipe fails compatibility CI rather than being silently skipped.
- Accept caller-created optimizer instances or optimizer factories. Forward native keyword arguments without reimplementing update algorithms. Schedulers may be supplied and run at accepted outer-step boundaries according to documented scheduler semantics.
- A closure clears gradients, recomputes the same deterministic full objective at the current trial parameters, accumulates batch gradients, and returns the scalar loss. L-BFGS may call it repeatedly. Do not count closure calls as accepted iterations or write checkpoints for rejected trial states.
- Keep source sets, weights, grid, stencil, PML and frequency stage fixed within an optimizer step. Repeated closure calls at identical parameters must produce equivalent values/gradients within the declared numerical tolerance.
- Differentiate compatibility tests from reconstruction benchmarks: test every native algorithm on valid small fixtures; demonstrate useful FWI convergence on selected Adam and L-BFGS runs, without asserting universal superiority or monotonicity.
- Optimizer state, scheduler state, parameterization, RNG state and accepted-step counters are restart state. Native optimizer weight decay is separate from the stated physical regularizer and must be visible in run metadata.

## Hessians and local uncertainty

Expose Jacobian JVP/VJP, Gauss–Newton action, and exact Hessian action in physical slowness-squared coordinates. For real model vectors:

    H_GN v = Re[J^H W^H W Jv] + H_R v.

The exact Hessian includes residual-dependent second-order wavefield/operator terms. At zero data residual, exact data Hessian and Gauss–Newton coincide; at nonzero residual they generally differ. Preserve real symmetry and support arbitrary real vectors. Explicit matrices are restricted to small diagnostics with a memory guard.

Curvature instances freeze a physical reference model and optional active-cell mask. Their vectors have shape [n_active], with explicit injection into and restriction from the full physical grid. Recreate the instance after changing the reference model. Provide a separate data-only Gauss–Newton action so uncertainty code can add a supplied prior exactly once; do not accidentally add the inversion regularizer and the same prior twice.

Expose theta-coordinate actions separately and label them. Exact transformed Hessians include the second derivative of the parameter transformation; they are not obtained by applying only the first-order chain rule. Sparse parameter optimizers may use physical-coordinate HVPs independently of native sparse higher-order autograd support.

Local Gaussian/Laplace uncertainty requires explicit noise whitening and a proper positive-definite prior precision (or a documented identifiable subspace). A gradient-only smoothness prior has a nullspace; do not silently treat it as invertible. Provide a positive diagonal prior term for the initial uncertainty fixture. Compute covariance inverse actions, seeded randomized diagonal estimates with sampling uncertainty, and prior-preconditioned low-rank/Lanczos diagnostics. Inverse diagonal is not reciprocal diagonal. Exact indefinite Hessians must not be inverted and labeled covariance without a justified positive-definite model.

## Output recording and restart

Use HDF5 for the initial local output backend; leave a small writer interface for future Zarr/distributed storage. Avoid implementing two storage backends before one is validated.

- Always record lightweight accepted-step diagnostics; checkpoint accepted model/optimizer state at a configurable interval (default every accepted outer step).
- Model snapshots default to checkpoint cadence; gradients are optional; wavefield output is off by default and selectable by frequency/source/step. Krylov bases are not archived by default.
- Separate compute buffers/checkpoints from archival recording. Detach recording payloads; never hold autograd graphs in callbacks.
- GPU transfers use reusable pinned buffers and events. A single writer owns the HDF5 handle. Default queue capacity is two output chunks, with backpressure, flush/join, and propagated writer errors. Bound bytes as well as queue item count.
- Chunk wavefields by frequency/source batch and spatial tile, selected from actual read patterns. Lossless compression is optional and benchmarked; default to uncompressed when compression slows writing without useful savings.
- Write checkpoints via temporary files plus an atomic rename/committed manifest on the supported local filesystem. Restart validates dataset/model/config hashes, schema version, parameterization, optimizer class and shapes. An interrupted write does not replace the last complete checkpoint.
- Include SI units, axes, origins, frequencies, physical/PML distinction, precision, source normalization, model identity, package versions, Git revision, seeds and solver tolerances in metadata.
- Lossy archival compression, if later added, is visualization-only until separately validated for derivative reuse. No zero-overhead I/O claim: measure throughput, peak host/device memory and total runtime with recording off/on.

## Dispersion assessment and Marmousi scaling

There is no time-stepping CFL condition in this frequency-domain solve. Distinguish spatial dispersion, linear-system conditioning/convergence, PML reflections, and nonlinear optimizer failures.

The assessment uses maximum frequency, minimum allowed velocity, grid spacings, stencil symbol and propagation direction. Initial spatial target: <=1% relative phase-wavenumber error over sampled propagation angles in a homogeneous medium. This is a local screening target, not a global heterogeneous-data accuracy guarantee. Validate using refined-domain data comparisons as well.

Estimate a practical ceiling from f <= v_min/(p h), with p derived from the selected stencil/tolerance. Optional automatic planning chooses only previously validated grid/stencil/PML configurations within a user memory budget. It returns a reviewable plan and estimated resources. Do not silently reduce requested accuracy or frequency. Fix the discretization inside each optimization stage and reset incompatible optimizer histories after regridding or changing the stage objective.

For Marmousi, begin with the provided 25 m data, then examine 12.5 m and 6.25 m computational grids only if the accuracy/resource plan allows. Interpolation does not add geological resolution. Candidate frequencies are 2, 5, 10, 15, 20 and 30 Hz; these bound the initial benchmark search, not the public API. Validate increasingly dense nested source sets up to one source per usable horizontal grid column. Record marginal information/runtime benefit, and distinguish this practical maximum from a physical upper bound on source count. Performance reports include PML cells, active cells, frequency/source count, tolerance, setup time, all RHS time, gradient time, HVP time, peak memory, I/O, GPU type and cost.

## Validation and acceptance gates

Thresholds below are initial fixture acceptance targets, not claimed measured results. Fix fixture definitions before collecting benchmark results. Any threshold change needs scientific justification recorded in the validation report; never relax a test just to obtain a pass.

1. Operator agreement: on small well-conditioned fixtures, sparse vs matrix-free relative action error <=1e-12 in complex128; interpolation and complex operator adjoint relative error <=1e-10.
2. Reference solves: true relative residual <=1e-10 for nonzero RHS, with absolute tolerance 1e-12 for zero/tiny RHS. Use stricter tolerances for derivative checks if needed to expose their convergence region.
3. Physics: second-order convergence trend on a smooth manufactured solution; homogeneous Green's-function/refined-reference agreement with explicit near-source exclusion; free-surface reflection has the expected pressure sign; rectangular grids and nonzero origins are tested.
4. PML: reflected amplitude <=1e-3 on calibrated homogeneous fixtures over the selected frequency band and incidence angles 0, 30 and 60 degrees, measured against a larger-domain reference. Sweep thickness independently from physical-grid refinement. Low-frequency and corner cases are explicit checks.
5. Derivatives: gradient Taylor remainder slope between 1.8 and 2.2 over at least three consecutive epsilon values above the numerical floor; exact HVP vs gradient central difference relative error <=1e-5 at an appropriate epsilon; GN action vs explicit weighted Jacobian relative error <=1e-8; Hessian symmetry error <=1e-8. Test regularization alone and combined objectives.
6. Precision: CPU/GPU complex128 waveform and physical-gradient relative agreement <=1e-7 on matched small fixtures. Complex64 candidates require waveform relative error <=1e-3 and physical-gradient relative error <=1e-2 against the reference, plus passing the relevant Taylor/inversion checks. Near-zero reference norms use declared absolute tolerances.
7. Optimizers: every native class in the pinned release has a valid executable recipe, finite gradients/parameters/state and a successful step; deterministic closure, sparse representation, shape restrictions, restart and bounds are exercised. Adam and L-BFGS Camembert runs show objective reduction without requiring every iteration to decrease.
8. Memory/storage: with fixed batch/cache/queue limits, resident wavefield working storage does not grow linearly with total sources/frequencies. Account separately for data and mandatory native optimizer state, which can grow with parameter count/history. Saved arrays round-trip; interrupted writes and failed writers cannot report a successful checkpoint.

Camembert baseline: 48 by 48 physical nodes, 10 m spacing, origin (0,0), centre (235 m,235 m), radius 60 m, background 1500 m/s and inclusion 2000 m/s. Use a central 16 by 16 active parameter window for routine derivative tests and 24 by 24 for extended diagnostics. Frequencies 5/10/15 Hz are candidates subject to dispersion checks; derivative consistency may use an under-resolved grid only when explicitly labeled algebraic rather than physical validation. Use eight surrounding sources and 32 surrounding receivers for diagnostic illumination, then surface acquisition separately. Pin their exact coordinates in a fixture. Keep the true circle fixed physically under refinement and distinguish interface geometry error from smooth manufactured-solution convergence.

For small uncertainty tests, explicitly form J, GN, exact H and the positive-definite local covariance; compare inverse actions and stochastic/low-rank approximations. Use fixed seeds and confidence-aware criteria for stochastic estimates. Use same-discretization data for algebraic derivatives and independent/refined data for reconstruction accuracy claims.

## Paid GPU experiment policy

Run CPU correctness gates before renting GPU time. Inspect existing RunPod access and active/billed resources without printing credentials. Before starting any billed resource, confirm current price, available credit, prior expenditure for this task, storage costs and a reliable stop/delete mechanism. If task expenditure cannot be bounded or attributed, do not launch.

Reserve at least USD 2 of the USD 10 ceiling for overhead, shutdown latency and storage. Authorize no more than USD 8 minus already incurred costs for the bounded compute experiment; lower this amount if the reserve is insufficient. Derive a wall-time limit from the actual rate and all billed resources, enforce it from outside the workload where available, and check expenditure during execution. Stop compute and remove only task-created billable resources after preserving required results. A completed Python process is not proof billing stopped. Verify shutdown through provider state and report final cost or explicitly unresolved metering lag.

Benchmark one feasible GPU with small accuracy-checked cases first. No full Marmousi inversion, broad hardware sweep, or expensive Hessian study is implicitly included in the USD 10 benchmark. If those do not fit, report the tested envelope rather than request an automatic budget increase.

## Delivery and review

Implement in the dependency order of the companion plan. Each milestone produces tested software and a recorded validation result. Numerical changes receive mathematical review before performance claims. The final release includes reproducible Camembert and Marmousi configurations, a native optimizer compatibility table, precision/dispersion reports, restart/output documentation, and an honest CPU/GPU performance comparison.

Rewrite the root README completely when the validated API is available. Target 350–600 words excluding code and illustration labels. Lead with what the package does, give separate CPU and GPU installation instructions, show one short executable Camembert example using a native PyTorch optimizer, and link to full examples and reference documentation. Include a compact workflow illustration (model/acquisition -> Helmholtz + PML -> data/objective -> adjoint gradient -> optimizer -> updated model) and a domain illustration showing physical cells, padded PML and the optional free surface. Prefer Mermaid for the workflow and a reproducibly generated SVG for the domain; provide readable alt text/captions. Illustrations must match actual array/geometry conventions. Mention matrix-free Hessian access and optional output recording briefly, with links rather than long explanations. State validated capabilities and limitations without unmeasured speedup claims, unlimited-scale promises, placeholder commands or broken links. Preserve clear license and citation/attribution links; do not invent a publication. README examples must run in CI against the public API. Save accompanying usage details in tracked Markdown outside the repository's ignored `docs/` path, or explicitly track individual intended documentation files without broadly changing ignore behavior.

Companion plan: [implementation plan](../plans/2026-10-15-fwi-redesign.md).
