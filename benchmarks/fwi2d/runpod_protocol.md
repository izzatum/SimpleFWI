# Bounded RunPod validation

The current implementation has **463 passing CPU tests** and **10 passing CUDA
cases** on an NVIDIA A40. The CPU/CUDA solver benchmarks, three-stage Marmousi
continuation and notebook 01's CUDA branch passed on source revision `abbfe1f`.
The final pod was deleted immediately after result retrieval. The [current run manifest](results/runpod-multiscale.json)
records actual allocation, source-transfer, failure, billing and deletion facts;
[earlier attempts](results/runpod-attempts.json) and the
[credential-binding retry](results/runpod-retry.json) remain historical evidence.

## Supported access and immutable source

The published credential binding now permits `api.runpod.io` and
`rest.runpod.io`. GraphQL and documented API-v2 inventory, billing, creation,
logging and deletion work. Earlier SSH/Jupyter transport failures and REST
Cloudflare denials do not establish a numerical CUDA failure.

Create-time `cmd` executes a small launcher; authenticated API-v2 Server-Sent
Events return its logs and checksum-framed results. Large embedded-source requests
were rejected with HTTP 413 before allocation. A small launcher successfully
allocated a pod and obtained logs by downloading source from an immutable public
GitHub commit. Its first run stopped at an inherited environment-variable check
before source download; the pod was immediately deleted.

The corrected launcher removes the inherited provider-key variable by name
before downloading or executing the workload. It never receives the local
provider credential. Public source downloads use normal verified HTTPS, fixed
byte counts and SHA-256, validated archive paths/types/size limits, and the
reviewed commit. No access-control or TLS bypass, silent CPU fallback, precision
reduction, or relaxed solver tolerance is permitted.

Use an isolated task virtual environment while preserving the image's verified
CUDA Torch installation. Global pip installation encountered a Debian package
without uninstall metadata. Check the actual interpreter, Torch version and
module path before and after dependency installation, run `pip check`, and bind
notebook execution to that same interpreter.

## Numerical gates

The reviewed numerical source is commit
`abbfe1fa7102580a725990e8de3a241c4874b354`. Its full CPU gate is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/fwi2d -m 'not cuda' -q
```

On an actual CUDA device, require the pinned Python 3.12 / PyTorch 2.12.0 release
and execute:

```bash
python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print(torch.__version__, torch.cuda.get_device_name(0))'
python -m pytest tests/fwi2d -m cuda -q -rs --junitxml=cuda-tests.xml
python benchmarks/fwi2d/solver.py --device cpu --quick --maxiter 120 --output solver-cpu.json
python benchmarks/fwi2d/solver.py --device cuda --quick --maxiter 120 --output solver-cuda.json
python examples/fwi2d/multiscale_marmousi.py --device cpu --quick --steps 3 --output multiscale-cpu.json
python examples/fwi2d/multiscale_marmousi.py --device cuda --quick --steps 3 --output multiscale-cuda.json
```

Require exactly ten executed CUDA cases with zero failures, errors or skips.
They cover precision/Taylor and inversion checks, resident multigrid state,
streamed source memory, pinned-buffer/event recording, and multiscale PML and
free-surface comparisons against sparse-direct CPU solves. Source-transfer
collection checks must select the same ten test identities as the full suite.

For the complex128 solver benchmark, require primal true relative residual at
most 1e-10 and every solve report finite, nonnegative and converged. Each report
must satisfy either the configured relative tolerance (1e-11) or absolute
tolerance (1e-13); this is a sufficient check for every right-hand side in a
batch. Keep the maximum relative residual across all solves as a measured value:
tiny derivative right-hand sides can meet the absolute tolerance while exceeding
the relative tolerance. A first batch checker incorrectly rejected that valid
case; its failed batch and ten successful CUDA tests remain in the history.

Require waveform and parameter-gradient relative errors at most 1e-7 and
exact-HVP relative error at most 1e-5. The benchmark differentiates in
`theta = log(m / m_reference)`; separate CUDA tests check physical-`m` gradients.
The Marmousi continuation must run 5×7, 9×13, 9×13
physical grids at `(2 Hz)`, `(2 Hz)`, `(2, 3 Hz)`, respectively, with three accepted
steps per stage. Compare all six initial/final stage losses against same-host CPU
results at relative error at most 1e-7, and require each final loss below its own
initial loss. Intermediate Adam steps need not decrease monotonically.

Preserve the primary results before optional notebook execution. Notebook 01
must retain its original code cells and report actual CUDA execution before its
GPU branch is described as tested. Algebraic agreement and a short crop inversion
do not establish continuum accuracy, full Marmousi recovery, high-frequency
scalability or GPU speedup.

The final run's maximum CPU/GPU stage-loss difference was 6.2163e-10 relative.
The quick example took about 1.04 s on CPU and 13.08 s on CUDA, with 17,673,728
peak allocated CUDA bytes. Each stage reduced data misfit, but final model RMSE
was 34.84 m/s versus 33.41 m/s for the constant initial model. This experiment
does not demonstrate improved recovery or faster GPU execution. The separate
headless GPU notebook artifact contains no embedded plots; the public notebook
keeps its reviewed CPU illustrations without mixing outputs from different runs.

## Budget and cleanup

The cumulative task ceiling is **USD 10**, including every allocation attempt,
setup, storage, shutdown delay and metering lag. Reserve **at least USD 2** for
shutdown and overhead; the budget does not reset between retries. Previously
attributed charges and unsettled allocations are covered by a conservative
prior-cost reservation, recorded in each launch envelope.

The planned hardware is one A40, quoted at USD 0.59/hour, with a USD 0.75/hour
compute guard and 40 GB ephemeral disk. External controllers are armed and their
heartbeats verified before creation. They enforce a 45-minute wall limit,
10-minute setup limit, collector liveness, and a conservative USD 1 account-spend
guard. With USD 0.25 reserved for prior costs and disk conservatively bounded at
USD 4/672 hours, the maximum launch envelope including the USD 2 reserve is
about USD 2.817. Recheck actual pricing, prior attributed spending, credit and
complete inventories before each allocation.

Use only the exact task resource name and acknowledged pod ID for cleanup.
Durably record an acknowledged ID before accepting a completed create; reconcile
uncertain outcomes without blindly retrying. Inventory or billing failure must
not prevent deletion of a known owned pod. On success or failure, retrieve
available evidence, issue DELETE, confirm GET 404, and verify two complete
ownership-filtered inventories no longer contain the pod. Ending Python or
merely stopping compute is insufficient.

Existing user-provided global storage is preserved. These validation launches
use `mounts={}` and create no persistent volume. Deletion of their pods removes
their ephemeral disks; it does not imply that all account storage is empty.
For retained output, follow the [persistent-storage guidance](../../guides/outputs.md#persistent-cloud-storage)
and account for its separate billing.

## Evidence and accounting

Keep source/dataset/harness hashes, UTC timestamps, actual runtime versions and
device, JUnit counts, numerical tolerances/errors, stage schedules, timing and
memory measurements. Keep task-owned pod IDs, rates, deadline/cleanup proof and
resource-specific billing. Account-wide spending is a conservative guard, not
attribution of unrelated charges to this task. A zero or incomplete billing
snapshot does not mean free usage; distinguish observed itemized amounts,
allocation-time estimates and unresolved settlement.

Do not publish credentials, request headers, private connection details,
unrelated resource metadata or raw environment dumps. Unknown or unexecuted
measurements remain explicitly unknown. CPU manifests retain their original CPU
scope and do not become GPU evidence after a later allocation.

Provider references: [Pod connections](https://docs.runpod.io/pods/connect-to-a-pod),
[pricing](https://docs.runpod.io/pods/pricing),
[billing](https://docs.runpod.io/accounts-billing/billing), and
[API-v2 schema](https://docs.runpod.io/api-reference-v2/openapi.json).
