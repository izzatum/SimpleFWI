# RunPod validation results

**GPU validation remains unrun; both earlier task-created pods were confirmed deleted.**
The [latest read-only retry](runpod-retry.json) created no resource: REST remains
blocked by Cloudflare 403/error 1010, and API account/billing return 401 after
publication removed `api.runpod.io` from the existing key's destination list.
Restore both provider hosts on that existing entry in environment settings,
then save and publish. No binding correction was saved by the configuration
tool, which requires editing the existing entry in the UI.
The [preflight manifest](runpod-preflight.json) records current and historical access; the
[attempt manifest](runpod-attempts.json) records two bounded A40 deployments and
provider-confirmed cleanup.

The first pod exposed SSH but its direct connection was refused and the official
SSH gateway was unreachable. Its undeclared web-terminal probe was inconclusive.
A local authenticated Jupyter upload/execute/download test then passed. The
second pod explicitly exposed `8888/http` with authenticated Jupyter, but the
correct proxy endpoint returned HTTP 403 / Cloudflare error 1010. Neither pod
executed numerical tests or the notebook GPU branch. No GPU timings, speedup,
CUDA memory measurements, or remotely observed software versions are claimed.

Both pods were terminated with DELETE, followed by GET 404 and repeated empty
ownership-inventory checks. Network/global-volume inventories are empty; no
persistent volumes were created. Account balance observations and a conservative
allocated-time cost estimate are recorded in the manifest. Final itemized billing
has not settled, so a zero-record billing response is not interpreted as free
compute. The latest blocked retry could not refresh inventory or settled billing;
the prior deletion confirmations remain historical evidence.

Follow the [bounded protocol](../runpod_protocol.md) before a future attempt. The task ceiling is USD 10 for every billed resource, with at least USD 2 reserved for shutdown/storage/metering and at most USD 8 minus prior task expenditure available for compute. Do not launch if costs or reliable shutdown cannot be bounded.

The existing [CPU smoke manifest](../marmousi-cpu-smoke.json) and [full-domain CPU manifest](../marmousi-full-cpu.json) are separate CPU observations, with their own accuracy limitations. They do not demonstrate GPU execution, GPU speedup, independent heterogeneous/PML accuracy or full inversion recovery.

A future executed run must preserve small structured evidence containing:

- Schema version, reviewed Git revision, configuration/dataset hashes, UTC times, seeds, Python/PyTorch/CUDA versions and GPU model/count.
- Physical/padded grid, SI units and axes, frequencies, source/receiver coverage, batch size, PML, tolerances, precision and memory limits.
- Actual test counts and skips, waveform/physical-gradient errors, Taylor/inversion evidence, solver true residuals, convergence status and explicitly labeled HVP errors.
- Setup/all-RHS/gradient/HVP timings and peak memory, compared at matched accuracy; distinguish estimates from measured device/host values.
- Task-owned resource IDs, verified rate/storage quote, prior attributed expenditure, credit/rate snapshots, deadline and external controller evidence.
- Result retrieval hashes, provider-confirmed final compute/storage state, attributed billed expenditure and explicitly unresolved metering lag.

Do not add credential values, request headers, authentication-bearing URLs, SSH private keys, raw environment dumps or large wavefield blobs. Unknown fields remain `null` or explicitly unknown; skipped/unexecuted work cannot be marked passed. The manifests are evidence of access, bounded attempts and cleanup, not executed GPU performance results.
