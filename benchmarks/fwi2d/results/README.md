# RunPod validation results

**GPU validation is blocked and unrun.** The [preflight manifest](runpod-preflight.json) records runtime HTTP failures and unknown accounting inputs. No GPU workload executed and no provider resource was created or modified. There are no GPU timings, speedup measurements, validated CUDA memory results or final task-billing totals to report.

The saved environment credential destinations include both provider hosts, but latest runtime requests returned HTTP 401 for account/pricing and HTTP 403 for REST inventory/billing. These responses do not prove the real key expired. Environment publication remains pending UI confirmation; no publication tool was available to this session. Current credit, exact rates, historical task spend and complete storage inventory remain unknown. The initial successful empty Pod list is historical and cannot establish today's total billed-resource inventory.

Follow the [bounded protocol](../runpod_protocol.md) before a future attempt. The task ceiling is USD 10 for every billed resource, with at least USD 2 reserved for shutdown/storage/metering and at most USD 8 minus prior task expenditure available for compute. Do not launch if costs or reliable shutdown cannot be bounded.

The existing [CPU smoke manifest](../marmousi-cpu-smoke.json) and [full-domain CPU manifest](../marmousi-full-cpu.json) are separate CPU observations, with their own accuracy limitations. They do not demonstrate GPU execution, GPU speedup, independent heterogeneous/PML accuracy or full inversion recovery.

A future executed run must preserve small structured evidence containing:

- Schema version, reviewed Git revision, configuration/dataset hashes, UTC times, seeds, Python/PyTorch/CUDA versions and GPU model/count.
- Physical/padded grid, SI units and axes, frequencies, source/receiver coverage, batch size, PML, tolerances, precision and memory limits.
- Actual test counts and skips, waveform/physical-gradient errors, Taylor/inversion evidence, solver true residuals, convergence status and explicitly labeled HVP errors.
- Setup/all-RHS/gradient/HVP timings and peak memory, compared at matched accuracy; distinguish estimates from measured device/host values.
- Task-owned resource IDs, verified rate/storage quote, prior attributed expenditure, credit/rate snapshots, deadline and external controller evidence.
- Result retrieval hashes, provider-confirmed final compute/storage state, attributed billed expenditure and explicitly unresolved metering lag.

Do not add credential values, request headers, authentication-bearing URLs, SSH private keys, raw environment dumps or large wavefield blobs. Unknown fields remain `null` or explicitly unknown; skipped/unexecuted work cannot be marked passed. The preflight manifest is evidence of a blocker, not an executed performance result.
