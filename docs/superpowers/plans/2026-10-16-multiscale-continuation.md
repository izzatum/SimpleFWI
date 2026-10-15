# Multi-resolution and multi-frequency continuation

The user requested implementation and GPU validation on 16 October 2026 (Malaysia time), within the existing cumulative USD 10 RunPod budget. Existing simultaneous multi-frequency objectives and single-stage grid screening already work. This change executes coarse-to-fine, low-to-high schedules using those established components.

## Design

- Explicit immutable stages specify grid, frequency tuple, iteration count, PML and source batch size. Single frequencies, cumulative frequencies and user-chosen bands share the same API.
- The initial velocity is supplied on the first/coarsest grid. Transfer detached physical squared slowness by bilinear interpolation at matching physical nodes. Preserve origin and node extent; reject coarsening. Interpolation adds no geological information. General anti-aliased model restriction is outside this change.
- Preserve acquisition locations/amplitudes in metres and select observed data and broadcast weights by frequency index. Do not interpolate measurements or normalize the summed objective implicitly.
- Preflight the entire schedule: positive available frequencies, nondecreasing frequency maxima, same physical domain/top boundary, nondecreasing physical PML widths, compatible precision/device and dispersion screening at the declared minimum velocity.
- Rebuild the operator, sampling, bounded model and native optimizer at every stage. Never transfer optimizer moments, L-BFGS history, unconstrained theta, masks or differentiation graphs. Explicit factories configure masks, regularization, optimizer and recording. Stage-local exact/GN Hessians remain available through the final objective/model.
- Keep only scalar histories and the final model; release obsolete solvers/caches. Stage-specific recording can archive intermediate models without retaining all stages on the GPU.
- Preserve all residual tolerances and precision/device choices. No GPU or large-scale speed claim without executed evidence. PML thickness reporting/dispersion screening do not replace calibrated physical validation.

## Implementation and validation

1. Add `multiscale.py`, public exports and meaningful transfer, schedule, frequency/objective, native optimizer, failure and CPU/CUDA parity tests.
2. Add a runnable Marmousi crop example with explicit CPU/GPU selection and a guide showing staged usage and interpretation.
3. Independently review the implementation; run the complete new 2D CPU suite and the example.
4. Once the CPU gate passes, use a bounded RunPod batch command and authenticated logs if available. Verify all selected CUDA tests execute, collect small numerical artifacts, and DELETE the task pod immediately afterward. Verify provider absence and storage cleanup. Include prior spend and at least USD 2 reserve in the USD 10 ceiling.
5. Record actual results, limits and exact commands. Report blocked/unrun work honestly.

## Decisions

The user explicitly authorized implementation; no additional design permission is required. Initial coarsening is deliberately explicit user preprocessing so the runner cannot silently alias a fine starting model. Each stage declares its PML because one scalar padding width cannot preserve both physical widths under arbitrary anisotropic refinement. Optimizer factories support native eager initialization (including Adagrad); fresh parameter ownership and histories are required.

## Status

Implementation and independent review are complete. The CPU gate passed 463 tests, with 10 CUDA cases deselected; a separate NVIDIA A40 run subsequently passed all 10 CUDA tests with zero failures, errors or skips. CPU/GPU solver and exact-HVP benchmarks passed, and all three Marmousi continuation stages reduced data misfit with maximum CPU/GPU stage-loss difference 6.2163e-10 relative. Notebook 01 also executed its actual CUDA branch successfully. These small runs do not establish reconstruction improvement, high-frequency scalability or GPU speedup; final crop model RMSE exceeded the constant initial model's RMSE.

Review corrections preserve frozen float32 bound values through transfer and validate scheduler ownership after stage configuration. The final task pod was deleted immediately after retrieving results, with independent GET 404 and two complete ownership-filtered absence checks. Existing global storage was preserved. The guides and benchmark manifests retain exact source/runtime, numerical results, prior failures and attributed cost estimates. The README now includes the new continuation workflow, direct-answer FAQ, license and traceable validation links. All tutorial CPU figures remain intact; the headless GPU notebook execution is a separate artifact.
