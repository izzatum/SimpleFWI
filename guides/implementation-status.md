# Implementation status — 15 October 2026

The new `simplefwi.fwi2d` workflow is implemented alongside the legacy API.
CPU numerical validation is available; GPU validation and high-frequency
Marmousi reconstruction remain incomplete. This is a research implementation,
not evidence of production-scale GPU performance.

| Capability | Delivered evidence and limits |
| --- | --- |
| Helmholtz physics | Conservative second-order operator, matching sparse CSC and matrix-free actions, padded coordinate-stretched PML, optional pressure-release top, SI geometry and complex adjoints. Manufactured convergence, analytic Green functions, free-surface sign and calibrated PML reflection tests pass. |
| Differentiable inversion | Implicit first and second derivatives, source-batch gradient accumulation, bounded velocity parameters, deterministic native optimizer closures. All 15 optimizer classes in PyTorch 2.12.0 have executable CPU recipes; flags and sparse representation constraints are documented. |
| Curvature and uncertainty | Physical and dense-parameter exact/Gauss–Newton Hessian actions, active-cell restriction, guarded explicit diagnostics, proper-prior local covariance actions, stochastic diagonal estimates and low-rank modes. Camembert reference comparisons support these claims; uncertainty remains conditional on the stated local Gaussian/noise/prior model. |
| Outputs and restart | Bounded HDF5 writer with backpressure, optional selected arrays, accepted-state callbacks and atomic native-state/RNG checkpoints. Restart validates the actual velocity bounds and representation. CUDA pinned buffers/events are implemented but have not run on hardware. Recording and checkpointing have measurable cost. |
| Dispersion and resource planning | Angular finite-difference symbol screening and immutable resource estimates. A returned `screened_only` plan is not physical validation. Automatic execution of regridding/stage transitions is not supplied; applications must construct a new stage and reset incompatible optimizer history. |
| Iterative solver | Restarted PyTorch FGMRES and shifted Galerkin multigrid pass CPU algebraic/derivative checks. The CPU benchmark found no speed crossover and two harder cases failed convergence. Sparse-direct CPU solves remain the reference default. |
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

## Verification and outstanding access

Run the new suite with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/fwi2d -q
```

The final run completed with **367 passed and 8 CUDA tests skipped** in 140.28 s.
Independent final review approved the CPU implementation with no unresolved
verified P0/P1/P2 finding after two lifecycle/restart repairs.

The locked Python 3.12 / PyTorch 2.12.0 CPU environment was installed into a fresh
virtual environment, a wheel was built, and its README example ran outside the
checkout. The repository-wide legacy collection still aborts on the pre-existing
BB inversion import; see [validation](validation.md). The new suite does not
replace or conceal that result.

The three tutorials subsequently executed all **30 default code cells** on CPU
with **16 saved figures**, independent headless execution and scientific/visual
review. They demonstrate Marmousi forward solves, Adam/RMSprop/L-BFGS workflows,
and Camembert exact/Gauss–Newton curvature with local uncertainty.

RunPod account/rate/inventory access now works through GraphQL and API v2.
Two bounded A40 attempts were made: direct SSH was unreachable on the first;
the second explicitly exposed authenticated Jupyter on HTTP port 8888, but its
proxy returned Cloudflare error 1010. The Jupyter transport had passed a local
upload/execute/download test. No remote numerical workload executed.
**Both pods were terminated/deleted and their absence verified; no persistent
storage remains.** Final itemized provider billing is still pending. The
[attempt manifest](../benchmarks/fwi2d/results/runpod-attempts.json) distinguishes
observed account debits, estimates, and final billing. The
[RunPod protocol](../benchmarks/fwi2d/runpod_protocol.md) records the USD 10
ceiling, at least USD 2 reserve, external deadlines, access blocker and cleanup.
CUDA tests and GPU benchmarks remain unrun; no GPU speedup is claimed.

## Implementation decisions

Work stayed in the existing isolated cloud checkout, as required by the setup
workflow. Agents implemented disjoint modules in parallel and the controller
integrated changes and commits. This avoided shared-index conflicts; interface
assumptions were checked through integration tests and independent review.
The CPU reference remains the default because the iterative scaling gate failed.
Further solver architecture work needs a measured design decision before any
claim of large-scale high-frequency performance.
