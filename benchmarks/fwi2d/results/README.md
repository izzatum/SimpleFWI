# Validation evidence

The [multiscale CPU record](multiscale-cpu.json) contains **463 passing CPU tests**
and both executed Marmousi continuation examples. Its ten deselected CUDA cases
are unrun in that artifact.

The [current RunPod manifest](runpod-multiscale.json) preserves actual source
transfer attempts, allocated pod lifecycles, attributed costs and cleanup.
The final NVIDIA A40 run passed all **10 CUDA tests with zero failures, errors or
skips**, both solver benchmarks, the three-stage Marmousi continuation and the
Helmholtz notebook's CUDA branch. All task pods were deleted; existing global
storage was preserved.

The final run used numerical source `abbfe1f`, Python 3.12.3,
PyTorch 2.12.0+cu130, CUDA 13.0 and driver 580.178.04:

- [Actual runtime](runpod-runtime.json) and [CUDA JUnit results](runpod-cuda-tests.xml).
- Matched complex128 solver benchmarks: [CPU](runpod-solver-cpu.json) and
  [CUDA](runpod-solver-cuda.json), including every residual and derivative check.
- Matched Marmousi continuation: [CPU](runpod-multiscale-cpu.json) and
  [CUDA](runpod-multiscale-cuda.json). All three stages reduced their own data
  misfit; the six initial/final loss comparisons differ by at most 6.2163e-10
  relative. Model RMSE did not improve, and this small CUDA run was slower.
- Notebook 01 [summary](runpod-notebook-01-summary.json) and
  [compressed actual GPU execution](runpod-notebook-01.ipynb.gz). All 11 code cells
  executed unchanged. The headless run has no embedded plots; the public tutorial
  retains its separately reviewed CPU illustrations without merging executions.

The final manifest records hashes for these artifacts. Observed itemized task
charges were USD 0.09631 at the recorded snapshot; the conservative cumulative
allocation estimate was USD 0.21837, with recent billing still pending. Neither
amount is represented as a settled final invoice.

Historical records remain available:

- [Earlier SSH/Jupyter attempts](runpod-attempts.json).
- [Credential-binding retry](runpod-retry.json).
- [Access and cleanup history](runpod-preflight.json).
- [CPU smoke](../marmousi-cpu-smoke.json), [full-domain CPU](../marmousi-full-cpu.json)
  and [CPU solver comparisons](../solver-cpu.json).

Follow the [bounded RunPod protocol](../runpod_protocol.md). The cumulative task
ceiling is USD 10, with at least USD 2 reserved for shutdown/storage/metering.
Delete owned GPU pods after success or failure, verify their absence, and preserve
existing user storage. Attribute spending by owned resource ID; distinguish
itemized observations from estimates and unsettled billing.

Evidence must record source/config/dataset hashes, actual device/runtime,
executed tests and skips, numerical errors and tolerances, stage schedules,
measured timing/memory, and provider-confirmed cleanup. Preserve failed cases.
Do not include credentials, unrelated resource metadata or large wavefield blobs.
Small algebraic and crop-workflow checks do not certify full Marmousi recovery,
heterogeneous/PML continuum accuracy, high-frequency scalability or GPU speedup.
