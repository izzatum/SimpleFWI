# SimpleFWI 2D PyTorch Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a validated 2D acoustic PyTorch FWI workflow with PML, optional free surface, native optimizer compatibility, efficient Hessian/uncertainty operations, bounded output recording, and a concise illustrated README.

**Architecture:** A new `simplefwi.fwi2d` package separates physical contracts, discrete operators, solver adapters, implicit autodiff, optimization and recording. An independent CPU sparse reference establishes correctness before a PyTorch matrix-free GPU path is benchmarked. Native optimizer adapters preserve the same physical objective across valid dense/sparse parameter representations.

**Tech Stack:** Python 3.12; released and exactly pinned PyTorch; NumPy/SciPy reference numerics; pytest; HDF5/h5py; Matplotlib for reproducible SVG diagrams. No paid resource is required to write or review this plan.

**Spec:** [2026-10-15-fwi-redesign.md](../specs/2026-10-15-fwi-redesign.md). Read it before executing any task. This is a consolidated plan requested by the user; it does not claim that the implementation or numerical targets have already been verified.

## Global Constraints

- Python 3.12 is the initial supported and tested runtime.
- PyTorch is the public tensor, parameter, optimizer, and autodiff framework; real model parameters use float64 by default and wavefields use complex128 by default.
- Pin an exact released PyTorch version and compatible dependencies when implementation begins; commit the resolved environment and test the optimizer inventory for that version. Do not depend on nightly releases or promise unknown future APIs.
- Use SI units internally, Fourier convention exp(-i omega t), array order [z, x], and explicit coordinate origins.
- Observed/predicted data axes are [frequency, source, receiver]; batched wavefields are [source_batch, z_padded, x_padded]. Axes and units are saved with outputs.
- Use the existing checkout. Cloud tasks are already isolated; do not create a Git worktree unless explicitly requested.
- Never disable assertions, silently suppress failed solves, or count skipped/unexecuted tests as passing.
- No zero-frequency Helmholtz solves, silent coordinate clipping, silent device fallback, or silent precision downgrade.
- Total RunPod expenditure for this task must not exceed USD 10, including compute, storage and other billed resources.
- Root README rewrite is required, with a 350–600 word target excluding code and illustration labels, an executable example and workflow/domain illustrations.

All numerical acceptance thresholds, physical assumptions, solver choices, PML conventions and storage policies in the spec apply to every task. Stage boundaries are scientifically meaningful review points, not permission to omit later requirements. Performance claims require equal-accuracy comparisons.

## Review Focus

1. Nonzero origins, unequal grid spacings, and out-of-domain acquisition: preserve physical coordinates and reject invalid locations; tests in tasks 1–2.
2. In-place optimizer mutation, repeated closures and invalid steps: invalidate numeric caches, maintain deterministic evaluations and avoid recording rejected states; tests in tasks 3 and 6.
3. Parameter/PML extension and nonlinear coordinate transformations: include their derivatives and label Hessian coordinates; tests in tasks 4 and 7.
4. Singular uncertainty precision and near-zero residual/RHS: diagnose nullspaces and use appropriate absolute tolerances; tests in tasks 3 and 8.
5. Writer failure, queue saturation and interrupted restart: bound memory, propagate errors and preserve the last complete checkpoint; tests in task 9.

## Delivery structure

Implement the following subprojects in order. Commit each completed task with its tests; do not merge a failed numerical milestone into the default API.

| Milestone | Tasks | Independently reviewable result |
|---|---|---|
| A — Verified forward physics | 1–3, 5 | CPU Helmholtz/PML solver with physical validation |
| B — Differentiable FWI | 4, 6 | PyTorch objective, bounded model, native optimizers |
| C — Curvature and uncertainty | 7–8 | Verified HVPs and small-model uncertainty references |
| D — Recording and restart | 9 | Bounded output pipeline with restart integrity |
| E — GPU and accuracy planning | 10–11 | Matrix-free GPU candidate and dispersion controls |
| F — Reproducible release | 12–14 | Marmousi envelope, budgeted evidence and illustrated README |

Dependency clarification: task 5 depends on tasks 1–3 and may be completed before task 4; task 6 requires both 4 and 5. Task 10 consumes operator/solver contracts and must also pass derivative tests from 4/7 before production use. Task 13 cannot start until CPU correctness gates pass.

### File and interface map

All paths below are proposed; create them only during implementation. Keep existing `simplefwi/core.py`, `misfit.py`, `optimization.py`, legacy tests and notebooks intact until the migration task.

| File under `simplefwi/fwi2d/` | Responsibility |
|---|---|
| `types.py`, `datasets.py` | Immutable configuration/result records, SI model imports |
| `acquisition.py`, `pml.py`, `operator.py` | Sampling/scatter, extension/stretching, matched discrete actions/assembly |
| `solvers/base.py`, `solvers/direct.py`, `solvers/krylov.py`, `solvers/multigrid.py` | Solver contract, CPU factors, PyTorch FGMRES and preconditioning |
| `autograd.py`, `objective.py`, `regularization.py` | Implicit solves and differentiation, physical objective |
| `parameters.py`, `optimizers.py`, `inversion.py` | Native parameter representations, optimizer recipes, accepted-step lifecycle |
| `curvature.py`, `uncertainty.py` | JVP/VJP, GN/exact Hessians, covariance inverse actions/estimation |
| `recording.py`, `checkpoint.py` | Detached bounded output, metadata, restart |
| `dispersion.py`, `planning.py`, `api.py` | Accuracy checks, fixed stage plans, public workflow |

Supporting files: `tests/fwi2d/` for real executable tests, `examples/fwi2d/` for reproducible experiments, `benchmarks/fwi2d/` for measured performance, `guides/` for tracked detailed documentation, `asset/` for tracked SVG illustrations. Update setuptools package discovery: its current explicit single-package list would omit the new subpackages from a wheel.

## Task 1: Establish contracts and reproducible package installation

**Files:** Create `simplefwi/fwi2d/{__init__,types,datasets}.py`, `tests/fwi2d/test_contracts.py`, `requirements/fwi2d-py312.lock`; modify `pyproject.toml` package discovery and add a dedicated dependency extra without removing legacy dependencies.

**Interfaces:** Define immutable `Grid2D(shape, spacing_m, origin_m)` (all tuples ordered z,x), `Acquisition(sources_m, receivers_m, amplitudes)` with coordinate pairs (z,x), `BoundaryConfig(top, pml_cells, sigma_max, alpha=0)`, `SolverConfig(method, rtol, atol, maxiter, restart, device, dtype, memory_budget_bytes)`, and `SolveReport`. `load_marmousi(path, *, velocity_unit) -> tuple[Grid2D, Tensor]` returns SI velocities. Define configuration hashing via serializable values, not object identity.

- [ ] Write `test_marmousi_units_and_origin` asserting 25 m spacing, x origin -200 m, expected velocity ranges after explicit unit conversion, and rejection of an omitted/invalid unit. Add rectangular-grid and NaN/nonpositive-velocity validation tests.
- [ ] Run `python -m pytest tests/fwi2d/test_contracts.py -q`; confirm failures reflect missing contracts, not unrelated imports.
- [ ] Implement the records/loaders; resolve a stable PyTorch release compatible with Python 3.12 and the intended platform, pin all development dependencies, and record the native optimizer inventory. Configure package discovery to include `simplefwi.fwi2d` and its solver subpackage.
- [ ] Run the tests, install from the lock in an isolated environment, build a wheel and import the new package outside the checkout. Confirm legacy imports still work.
- [ ] Commit only the contract, packaging and test files with `feat: define SI grid and acquisition contracts`.

## Task 2: Implement matched sparse and tensor PML operators

**Files:** Create `simplefwi/fwi2d/{acquisition,pml,operator}.py`; tests `test_acquisition.py`, `test_operator.py`, `test_pml_extension.py` under `tests/fwi2d/`.

**Interfaces:** `HelmholtzOperator(grid: Grid2D, boundary: BoundaryConfig)` provides `apply(m: Tensor, omega: float, u: Tensor, *, adjoint: bool=False) -> Tensor` and `assemble(m, omega) -> scipy.sparse.csc_matrix`. Its active computational DOF map excludes Dirichlet nodes. `SamplingOperator(grid, positions_m)` provides `sample(u)` and `scatter(values)` with exact discrete adjointness. `extend_model(m, grid, boundary)` is a differentiable nearest-boundary map into PML; source normalization is explicit cell-area scaling.

- [ ] Write action-equivalence tests at <=1e-12 relative error and complex dot tests at <=1e-10. Add nonzero-origin, unequal-spacing, PML-corner, edge-cell extension derivative, free-surface-node rejection and no-coordinate-clipping tests.
- [ ] Run `python -m pytest tests/fwi2d/test_acquisition.py tests/fwi2d/test_operator.py tests/fwi2d/test_pml_extension.py -q` and confirm the missing implementation failures.
- [ ] Implement conservative second-order face/centre coefficients and cubic sigma in padding, ordinary PML first (alpha=0), top free-surface elimination, source/sampling operators and both operator representations using the same documented mathematical coefficients.
- [ ] Rerun the task tests; compare against a separately written small explicit stencil fixture so two shared implementations cannot conceal the same indexing error.
- [ ] Commit with `feat: add conservative Helmholtz operator with PML and free surface`.

## Task 3: Build CPU reference solves and bounded reusable state

**Files:** Create `simplefwi/fwi2d/solvers/{__init__,base,direct}.py`; `tests/fwi2d/test_direct_solver.py`, `test_solver_cache.py`.

**Interfaces:** `LinearSolver.solve(operator, m, omega, rhs, *, adjoint=False) -> tuple[Tensor, SolveReport]`; `DirectSolver(config: SolverConfig)` implements it. `SolverFailure` includes the report. Evaluation-scoped cache keys include the values/configuration specified in the spec; cache entries have byte estimates and an explicit lifecycle.

- [ ] Test forward/adjoint solves against dense `torch.linalg.solve` on tiny fixtures, <=1e-10 true relative residual, zero RHS with absolute tolerance 1e-12, repeated RHS factor reuse, bounded cache eviction, in-place model mutation and changed-frequency/PML invalidation. Assert an unconverged result raises before use.
- [ ] Run `python -m pytest tests/fwi2d/test_direct_solver.py tests/fwi2d/test_solver_cache.py -q` and confirm intended failures.
- [ ] Implement CPU CSC factors with batched RHS and supported conjugate-transpose solves; convert explicitly at the adapter boundary. Reject CUDA tensors on the CPU adapter rather than silently transferring them.
- [ ] Rerun tests and record factorization count and memory for one versus multiple source batches. Verify standalone wheel use, not just checkout imports.
- [ ] Commit with `feat: add validated reusable CPU Helmholtz solves`.

## Task 4: Implement implicit first and second derivatives

**Files:** Create `simplefwi/fwi2d/{autograd,regularization,objective}.py`; `tests/fwi2d/test_implicit_autograd.py`, `test_regularization.py`, `test_objective.py`.

**Interfaces:** `implicit_solve(m, rhs, omega, operator, solver) -> Tensor`; `Regularizer(grid, beta, prior_m).value(m) -> real scalar Tensor`; `FWIObjective(operator, acquisition, frequencies_hz, observed, weights, regularizer, solver)` provides `loss(m) -> real scalar Tensor` and `iter_data_losses(m, source_batch_size)` yielding source/frequency contributions. Define the loss sum once: the regularizer is added exactly once after batch accumulation.

- [ ] Write PyTorch `gradcheck` and `gradgradcheck` against tiny dense differentiable solves. Test complex RHS gradients, physical-boundary/PML extension derivatives, combined vs separate regularization gradients, and equivalence of batched/unbatched objective and gradients.
- [ ] Run `python -m pytest tests/fwi2d/test_implicit_autograd.py tests/fwi2d/test_regularization.py tests/fwi2d/test_objective.py -q`; confirm missing functionality fails.
- [ ] Implement implicit adjoint backward and differentiable backward recomputation for second derivatives. Do not differentiate through Krylov iterations or detach required higher-order dependencies. Implement correct half-factor spatial regularization and weighted residuals.
- [ ] Rerun tests, require gradcheck/gradgradcheck PASS, and perform epsilon-sweep Taylor checks with slopes 1.8–2.2 over at least three usable points. Test a nonzero residual and a nonlinear model perturbation, not only a perfect fit.
- [ ] Commit with `feat: support verified implicit Helmholtz autodiff and objective`.

## Task 5: Establish independent Camembert physics fixtures

**Files:** Create `examples/fwi2d/camembert.py`, `tests/fwi2d/fixtures/camembert.json`, `test_physics.py`, `test_pml_reflection.py`, `test_camembert.py`, and `guides/validation.md`.

**Interfaces:** `make_camembert(config_path) -> tuple[Grid2D, Tensor, Tensor, Acquisition, BoundaryConfig]` returns true/initial velocity and a fixed acquisition. Pin the spec's 48 by 48 grid, circular inclusion and candidate frequencies. Use surrounding acquisition on the perimeter of the interior square [40,430] m with eight sources and 32 distinct receivers; save exact coordinate arrays. Add a separate surface-acquisition fixture with sources at z=40 m and receivers at z=60 m.

- [ ] Write smooth manufactured-solution refinement tests, homogeneous field/source normalization tests excluding the source neighborhood, pressure-free-surface reflection-sign tests, and PML reflection sweeps with <=1e-3 amplitude at 0/30/60 degrees on calibrated fixtures. Include low-frequency and corner cases.
- [ ] Run `python -m pytest tests/fwi2d/test_physics.py tests/fwi2d/test_pml_reflection.py tests/fwi2d/test_camembert.py -q`; verify failures expose absent evidence or incorrect physics.
- [ ] Implement fixture generation and independent references: analytical homogeneous Green's functions and a larger/finer computational reference for Camembert. Keep circle geometry fixed physically under refinement. Classify same-grid derivative data separately from independent reconstruction data.
- [ ] Rerun tests, store small scalar convergence/reflection summaries and reproducible commands in `guides/validation.md`. If a candidate frequency fails dispersion, label it algebraic-only until task 11 provides a physically acceptable configuration; do not claim a physics pass.
- [ ] Commit with `test: establish independent Camembert and PML validation`.

## Task 6: Integrate all native optimizer families and the accepted-step lifecycle

**Files:** Create `simplefwi/fwi2d/{parameters,optimizers,inversion}.py`; `tests/fwi2d/test_parameters.py`, `test_optimizer_inventory.py`, `test_closure.py`, `test_inversion.py`; `guides/optimizers.md`.

**Interfaces:** `VelocityParameters(initial_velocity, bounds, active_mask, *, representation="dense")` is an `nn.Module` with `velocity()`, `slowness_squared()` and native `parameters()`. Dense parameters are 2D; `representation="sparse_embedding"` uses sparse-gradient embeddings. `OptimizerRecipe` records class, valid construction arguments, representation and constraints. `InversionRunner(objective, model, optimizer, *, source_batch_size, scheduler=None, recorder=None)` provides `step() -> StepResult` and `run(steps) -> InversionResult`; StepResult distinguishes accepted outer steps from closure evaluations.

- [ ] Write a discovered-inventory test requiring an executable recipe for each stable native optimizer class in the pinned release, including SparseAdam and any shape-constrained methods. Assert finite parameter/state values after one valid step, physical bounds, unchanged frozen cells and exact physical-objective equivalence between dense/embedding representations. Explicitly reject incompatible direct optimizer/parameter combinations.
- [ ] Add closure tests: repeated calls at fixed parameters agree, gradients do not accumulate accidentally, LBFGS trial points invalidate caches, batch backward matches full loss, and only accepted states emit recording/scheduler events. Test failed solve propagation without a claimed successful step.
- [ ] Run `python -m pytest tests/fwi2d/test_parameters.py tests/fwi2d/test_optimizer_inventory.py tests/fwi2d/test_closure.py tests/fwi2d/test_inversion.py -q` and confirm the expected failures.
- [ ] Implement native parameter mappings/recipes and the runner. Stream first-order batch backward; do not hold all batch graphs. Keep closure source/frequency sets deterministic. Identify the actual final accepted parameters before recording their loss/state, including cases where the optimizer returns a pre-step loss.
- [ ] Rerun tests; run small Adam and LBFGS Camembert inversions showing objective reduction without asserting per-step monotonicity for Adam. Document tested native flags and unsupported combinations without claiming all flags work on every device.
- [ ] Commit with `feat: integrate native PyTorch optimizers with deterministic FWI closures`.

## Task 7: Provide physical and transformed Hessian operators

**Files:** Create `simplefwi/fwi2d/curvature.py`; `tests/fwi2d/test_jacobian.py`, `test_hessian.py`, `test_hessian_coordinates.py`; `examples/fwi2d/camembert_hessian.py`.

**Interfaces:** `Curvature(objective, m, *, active_mask=None)` freezes the reference model and provides `jvp(v)`, `vjp(data_vector)`, `data_gn_hvp(v)`, `gn_hvp(v)`, `exact_hvp(v)` in physical m coordinates. Model vectors have shape [n_active] with explicit injection/restriction; the default mask selects all physical cells. `data_gn_hvp` excludes regularization, while `gn_hvp` and `exact_hvp` include it. `TransformedCurvature(objective, model)` exposes separately labeled theta-coordinate actions including exact transformation second derivatives. `explicit_matrix(action, n, *, max_bytes)` is diagnostics-only and refuses allocations above the provided budget.

- [ ] Write tests comparing JVPs with data finite differences, GN with an explicit weighted Jacobian (<=1e-8 relative error), exact HVP with gradient central differences (<=1e-5), and real Hessian symmetry (<=1e-8). Check equality of data GN/exact Hessian at zero residual and the expected nonzero residual curvature difference.
- [ ] Test nonlinear theta transformation against `torch.autograd.functional.hessian` on a tiny dense reference, including its second derivative term; test physical HVPs for the sparse parameterization independently of sparse native double-backward support.
- [ ] Run `python -m pytest tests/fwi2d/test_jacobian.py tests/fwi2d/test_hessian.py tests/fwi2d/test_hessian_coordinates.py -q` and confirm intended failures.
- [ ] Implement operator actions with incremental forward/adjoint solves and explicit memory policy. Recompute/checkpoint source state rather than allocating J or H by default. Add the 16 by 16/24 by 24 active-window Camembert experiment.
- [ ] Rerun tests and `python examples/fwi2d/camembert_hessian.py --fixture tests/fwi2d/fixtures/camembert.json`; record all error metrics, coordinates, residual level and solve tolerances.
- [ ] Commit with `feat: add verified Gauss-Newton and exact Hessian actions`.

## Task 8: Implement local uncertainty with explicit statistical assumptions

**Files:** Create `simplefwi/fwi2d/uncertainty.py`; `tests/fwi2d/test_uncertainty.py`; `examples/fwi2d/camembert_uncertainty.py`; `guides/uncertainty.md`.

**Interfaces:** `LocalGaussian(curvature, prior_precision, *, solve_config)` builds precision from `curvature.data_gn_hvp` plus the supplied prior exactly once, and provides `covariance_action(v)`, `diagonal_estimate(probes, seed) -> Estimate` (mean and sampling-error metadata), and `leading_modes(rank, seed)`. Inputs and results name physical parameter coordinates and noise whitening. Require a positive prior diagonal for the initial fixture in addition to optional smoothing.

- [ ] Write small explicit-reference tests for inverse actions (<=1e-8 relative error on well-conditioned fixtures), seeded diagonal estimates with documented Monte Carlo error, and increasing-rank approximation checks. Assert singular/improper precision and an indefinite exact Hessian are rejected as covariance models. Test that inverse diagonal differs from reciprocal precision diagonal on a coupled fixture and that the prior is not double-counted when the inversion objective already has regularization.
- [ ] Run `python -m pytest tests/fwi2d/test_uncertainty.py -q` and confirm intended failures.
- [ ] Implement iterative SPD inverse actions and seeded stochastic/Lanczos estimators, reporting inner-solve errors separately from sampling/truncation errors. No silent damping or pseudoinverse masquerading as a proper posterior.
- [ ] Rerun tests and the Camembert example; compare with explicit small covariance and document the local Gaussian approximation's limits.
- [ ] Commit with `feat: add reference-validated local uncertainty operators`.

## Task 9: Add bounded recording and trustworthy restart

**Files:** Create `simplefwi/fwi2d/{recording,checkpoint}.py`; `tests/fwi2d/test_recording.py`, `test_restart.py`; `guides/outputs.md`.

**Interfaces:** `RecordingPolicy` specifies output path, selected fields/frequencies/sources, cadence, chunk byte limit, queue capacity (default two), and compression. `Recorder.record_accepted(step_result, payload)` accepts detached arrays and `close()` flushes or raises. `save_checkpoint(path, state)` and `load_checkpoint(path, expected_config)` include model/optimizer/scheduler/RNG and metadata. Payload names distinguish physical/PML wavefields and physical/parameter-coordinate gradients.

- [ ] Write round-trip and interrupted-write tests; inject writer errors and slow writers, assert a two-chunk/byte-bounded queue, backpressure, and no retained autograd graphs. Assert rejected closure evaluations never persist as accepted snapshots and partial writes cannot overwrite the last complete checkpoint.
- [ ] Test restart equivalence on a deterministic small optimizer run, mismatched parameterization/optimizer/config rejection, source-batch selections and complete axis/unit metadata.
- [ ] Run `python -m pytest tests/fwi2d/test_recording.py tests/fwi2d/test_restart.py -q` and confirm intended failures.
- [ ] Implement one HDF5-owning writer, reusable buffers, atomic checkpoint/manifest publication, propagated errors and explicit lifecycle cleanup. Add GPU pinned buffers/events when task 10 provides CUDA execution. Default wavefields/gradients off; keep diagnostic records small.
- [ ] Rerun tests and benchmark recording off/on with bounded source batches. Report I/O cost and host/device peak memory without a zero-overhead claim.
- [ ] Commit with `feat: add bounded scientific output recording and restart`.

## Task 10: Implement and validate the PyTorch production solver

**Files:** Create `simplefwi/fwi2d/solvers/{krylov,multigrid}.py`; `tests/fwi2d/test_krylov.py`, `test_multigrid.py`, `test_device_precision.py`, `test_memory_scaling.py`; `benchmarks/fwi2d/solver.py`.

**Interfaces:** `FGMRESSolver(config, preconditioner)` implements the task 3 solve interface. `ShiftedHelmholtzMultigrid(operator, config).apply(m, omega, rhs, *, adjoint=False)` provides a bounded preconditioner; its shift and level policies are explicit configuration/benchmark output, not hidden defaults. Support tensor CPU execution first, then CUDA without changing the physical operator.

- [ ] Write CPU tensor tests against DirectSolver, explicit nonconvergence tests, adjoint solve checks, compatible coarse-grid boundary tests, and restarted basis-memory bounds. Add CUDA-marked tests for complex128/complex64 thresholds from the spec; absent hardware is reported as unrun, never validated.
- [ ] Run `python -m pytest tests/fwi2d/test_krylov.py tests/fwi2d/test_multigrid.py -q` to establish the failing baseline.
- [ ] Implement restarted FGMRES with true-residual verification and shifted-operator multigrid; profile tensor allocations before adding custom kernels. Test preconditioner choices on increasing frequency/contrast and stop claiming scalability if iterations grow beyond the declared budget.
- [ ] Rerun CPU tests plus task 4/7 derivatives through the new solver. When authorized bounded GPU runtime is available under task 13, execute device/precision/memory tests, verify cached data stay on the selected device, and test recording transfers from task 9.
- [ ] Benchmark setup plus all RHS, forward/gradient/HVP wall time, residuals and memory against direct solves. Report the crossover and unresolved cases; do not automatically favor GPU.
- [ ] Commit with `feat: add accuracy-checked PyTorch matrix-free Helmholtz solver`.

## Task 11: Add dispersion screening and fixed-stage resource planning

**Files:** Create `simplefwi/fwi2d/{dispersion,planning}.py`; `tests/fwi2d/test_dispersion.py`, `test_stage_plan.py`; `guides/accuracy.md`.

**Interfaces:** `assess_dispersion(grid, velocity_bounds, frequencies_hz, stencil) -> DispersionReport`; `plan_stage(request, *, memory_budget_bytes) -> StagePlan` returns a fixed grid, stencil, frequency set, source batching, precision and PML configuration plus estimates and validation status. Implement the validated second-order stencil first; an additional optimized/high-order stencil must pass task 2/5/7 checks before registration.

- [ ] Write analytic-symbol angular tests with <=1% phase-wavenumber error target, rectangular grids, minimum allowed velocity rather than only current velocity, and explicit failure when no configuration meets the accuracy/memory request. Verify zero frequency rejection and stage immutability during optimizer steps.
- [ ] Run `python -m pytest tests/fwi2d/test_dispersion.py tests/fwi2d/test_stage_plan.py -q` and confirm intended failures.
- [ ] Implement assessment and bounded selection over validated configurations. Treat PML reflection and linear solve convergence as independent gates. Refinement is allowed at stage boundaries only; map physical models consistently and reset incompatible optimizer histories.
- [ ] Rerun tests and compare screened configurations with refined-reference wavefields. Record that local homogeneous dispersion estimates are not a global heterogeneous accuracy guarantee.
- [ ] Commit with `feat: add explicit dispersion and memory-aware stage planning`.

## Task 12: Publish reproducible Marmousi experiments and public API

**Files:** Create `simplefwi/fwi2d/{api,__main__}.py`, `examples/fwi2d/marmousi.py`, `examples/fwi2d/configs/{camembert,marmousi}.json`, `tests/fwi2d/test_public_api.py`; update new package exports only.

**Interfaces:** `AcousticFWI(grid, acquisition, boundary, solver_config)` exposes `simulate(velocity, frequencies_hz, *, source_batch_size, recorder=None)` and `objective(observed, frequencies_hz, *, weights=1.0, regularizer=None) -> FWIObjective`. Users construct `VelocityParameters`, their native optimizer and `InversionRunner`; convenience APIs must preserve those explicit objects. A CLI accepts the same serialized experiment configuration and emits a run manifest.

- [ ] Write a tiny end-to-end public-API test covering simulation, a native optimizer step, a physical HVP and output reload, plus invalid shape/unit/device errors. Test packaged example access outside the repository root.
- [ ] Run `python -m pytest tests/fwi2d/test_public_api.py -q` and confirm intended failures.
- [ ] Implement exports/CLI/examples using only public interfaces. Configure explicit SI imports, 25/12.5/6.25 m candidate grids, 2/5/10/15/20/30 Hz candidate frequencies and nested source sets to the usable aperture limit. Every expensive case requires a successful accuracy/resource preflight; default quickstart runs small Camembert, not the entire sweep.
- [ ] Run a small CPU Marmousi forward/gradient case and save its accuracy, resource and source-coverage report. Clearly mark larger or higher-frequency cases as planned/unrun where budgets prevent validation.
- [ ] Commit with `feat: expose reproducible Camembert and Marmousi workflows`.

## Task 13: Execute a bounded RunPod validation experiment

**Files:** Create `benchmarks/fwi2d/runpod_protocol.md`, `benchmarks/fwi2d/results/README.md`, and small structured result manifests (no secrets or large wavefield blobs).

**Interfaces:** Benchmark commands consume task 12 configs and emit the same versioned manifests; provider access remains outside the numerical library.

- [ ] Verify CPU correctness gates, existing access, current task expenditure, active resources, actual rates, available storage and stop/delete support. Do not create resources if the USD 10 total cannot be bounded. No tokens/credentials in logs, configs or commits.
- [ ] Write the concrete experiment envelope: one GPU, exact workloads, maximum wall time, cost calculation, shutdown reserve >=USD 2, compute allowance <=USD 8 minus incurred costs, and a cleanup plan for task-created resources only. Include metering lag and all billed resources.
- [ ] Start or reuse a suitable authorized resource within that envelope, enforce an external stop deadline where supported, and run task 10 device/precision tests plus one or more preflight-approved solver/gradient/HVP benchmarks. Do not install speculative alternate stacks or run an open-ended sweep on the meter.
- [ ] Preserve result manifests, verify provider-side stopped/deleted state for task-created resources, and check billed expenditure. Report unknown final metering as unknown; do not claim a completed script stopped billing.
- [ ] Commit sanitized evidence with `bench: record budgeted GPU validation and cost`. If access, budget or solver gates prevent a benchmark, record the blocker and leave GPU validation explicitly incomplete; never fabricate performance.

## Task 14: Rewrite the README and finalize release documentation

**Files:** Rewrite `README.md`; create `asset/fwi-workflow-domain.svg`, `examples/fwi2d/render_domain.py`, `guides/{quickstart,architecture,migration}.md`, `tests/fwi2d/test_readme_example.py`; update the compatibility matrix and validation guides from earlier tasks. Update CI to run actual new tests and distinguish GPU-required jobs.

**Interfaces:** README code uses only task 12 public interfaces and the locked installation path. Workflow illustration nodes correspond to actual public components; the domain SVG is generated from the same boundary/grid configuration shown in the example.

- [ ] Draft a concise outline: purpose; workflow illustration; CPU/GPU installation; runnable Camembert example with a native optimizer; PML/free-surface domain illustration; short Hessian/output usage links; measured capabilities/limitations; license/citation links. Target 350–600 words excluding code/illustration labels.
- [ ] Write `test_readme_example` to execute the fenced quickstart code in a temporary output directory, checking finite data/loss and successful optimizer execution. Add local-link existence checks and assert that the referenced SVG exists. Run `python -m pytest tests/fwi2d/test_readme_example.py -q` to establish missing-doc/example failures.
- [ ] Completely rewrite the root README around the validated implementation. Use Mermaid for the workflow and a deterministic Matplotlib SVG for the PML/free-surface geometry, with alt text/captions. Show one small example and link to advanced guides; omit unmeasured speedups, roadmap clutter, invented citations and unsupported backend claims.
- [ ] Generate the illustration, inspect it visually, execute README commands from a clean locked install, check rendered Markdown/Mermaid readability and links, and run the CPU test suite. Document the actual optimizer/device matrix, all known limitations and GPU checks that remained unrun. Do not hide unresolved legacy failures by calling the entire repository green.
- [ ] Write migration guidance from legacy dictionaries/flattened arrays to explicit SI grids and data axes. Preserve legacy entry points for this release unless a separate breaking-change decision is approved. Update package docs and CI to install/check subpackages from a wheel.
- [ ] Commit with `docs: rewrite README with verified illustrated FWI quickstart`. Release only capabilities with passing evidence; the presence of the plan is not a release certification.

## Completion and handoff

The plan is complete when its documentation captures all agreed requirements; implementation is complete only when the applicable task checkboxes and scientific gates have evidence. Do not mark GPU scaling, full Marmousi inversion, or uncertainty accuracy complete based solely on small CPU tests.

Before executing, review the specification and plan together and select the execution approach. Native sequential execution is a reasonable default because the operator/autograd/optimizer interfaces are tightly coupled; independent numerical review at milestone boundaries remains important. No agents, software changes or paid resources are started by saving this plan.
