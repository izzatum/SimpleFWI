# Bounded RunPod validation protocol

Task 13 is **blocked; GPU validation is incomplete**. No GPU workload ran and no provider resource was created, started, stopped, changed or deleted during this preflight. The versioned [preflight manifest](results/runpod-preflight.json) records observed access failures; it contains no fabricated performance measurements. [Results guidance](results/README.md) describes the evidence still required.

## Observed access, 15 October 2026

| Check | Observed result |
| --- | --- |
| Initial REST Pod list | HTTP 200, empty list |
| Latest REST Pod list and Pod billing retry | HTTP 403, `error code: 1010` |
| REST storage/serverless inventory and billing | HTTP 403 |
| Latest GraphQL account, Pod and GPU-price queries | HTTP 401 with both attempted authentication mechanisms |
| Latest REST API v2 GPU catalog | HTTP 401 |
| Credit, account hourly rate, live GPU quote, storage rates and prior task costs | Unknown |

The saved credential configuration now allows `api.runpod.io` and `rest.runpod.io`. Working runtime authentication has not followed from that metadata change. The environment's Publish UI action is pending user confirmation; this session had no callable publication tool and does not claim publication. The HTTP responses do not prove the real provider key expired or lacks permissions. No duplicate secret is required merely to repeat this diagnosis. The precise REST denial source is unverified; no proxy or network-policy bypass was attempted.

An empty historical Pod list does not establish an empty current account, zero historical expenditure, or zero separately billed network-volume/serverless storage. The preflight allocated USD 0 in new resources; total prior task cost and final total remain unknown.

## Required account and CPU gates

Before any billed allocation, confirm the exact reviewed revision and passing CPU numerical gates, then verify current credit, full active/billed resource inventory, historical task expenditure, deployable GPU availability and actual rates. The CPU verification command is:

```bash
python -m pytest tests/fwi2d -m 'not cuda' -q
```

This document does not assert that command passed; use the root task's actual validation record. Failed solver/physics/derivative gates must be resolved before paid performance work. All true-residual checks and default float64/complex128 precision remain active.

Documented inventory/accounting routes include:

- `GET https://rest.runpod.io/v1/pods`
- `GET https://rest.runpod.io/v1/networkvolumes`
- `GET https://rest.runpod.io/v1/endpoints`
- `GET https://rest.runpod.io/v1/billing/pods`
- `GET https://rest.runpod.io/v1/billing/networkvolumes`
- `GET https://rest.runpod.io/v1/billing/endpoints`
- `GET https://api.runpod.io/v2/catalog/gpus`
- Read-only GraphQL queries at `https://api.runpod.io/graphql` for account credit/current spend and actual GPU availability/pricing.

Billing windows must use explicit UTC start/end times, hourly buckets and resource grouping; attribute task costs to resource IDs instead of subtracting an unexplained account balance. Credentials stay in the secure environment/proxy binding. Do not print request headers, authentication URLs, environment dumps, tokens or private keys, or place them in manifests.

## USD 10 experiment envelope

Total RunPod expenditure for this task must not exceed **USD 10**, including compute, startup/download time, all disks/volumes, stopped storage, overhead, shutdown delay and metering lag. Reserve **at least USD 2** for shutdown, storage and accounting uncertainty. Authorize compute expenditure of **at most USD 8 minus already incurred task costs**; reduce it further when overhead or uncertainty exceeds the reserve.

Let `C_prior` be confirmed historical task cost, `R_all` the conservatively bounded hourly rate for every task-created billable resource, `C_fixed` any separately bounded startup/minimum-billing/retained-storage costs, and `C_reserve >= 2` the shutdown/storage/metering reserve. The concrete deadline must satisfy:

```text
C_prior + C_fixed + R_all * T_hours + C_reserve <= USD 10
compute_allowance <= USD 8 - C_prior
```

There is **no currently approved numeric wall time or selected GPU rate**: those facts are unknown. Do not create/start a resource until every term, sufficient available credit and a reliable external stop/delete mechanism are verified. The provider's default account hourly spend limit does not enforce this task's USD 10 total ceiling. Existing unrelated resources remain outside cleanup ownership; monitor their credit consumption without inventing task attribution.

Public reference storage prices are USD 0.10/GB/month for running container/volume disks, USD 0.20/GB/month for stopped Pod volume disks and USD 0.07/GB/month for standard network volumes below 1 TB. These are documentation, not a verified allocation quote. Container/Pod-volume storage is billed per second and network volumes hourly according to the detailed pricing page. The general billing page describes a five-minute billing cycle. Account for minimum units and lag conservatively; confirm actual selected-resource prices before launch.

## Exact initial workload and commands

Use one GPU, Python 3.12 and the repository's exact supported PyTorch pin. First verify the CUDA device exists, then run task-10 marked precision/device tests and one fixed small algebraic solver/gradient/exact-HVP benchmark. Commands below are **planned, unexecuted** and require the preceding gates plus external provider shutdown enforcement:

```bash
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"; assert torch.cuda.device_count() == 1, "one GPU required"; print(torch.__version__, torch.cuda.get_device_name(0))'
python -m pytest tests/fwi2d -m cuda -q -rs
python benchmarks/fwi2d/solver.py --device cpu --quick --maxiter 120 --output benchmarks/fwi2d/results/solver-cpu-complex128.json
python benchmarks/fwi2d/solver.py --device cuda --quick --maxiter 120 --output benchmarks/fwi2d/results/solver-cuda-complex128.json
```

Run the CPU comparison before GPU allocation when feasible. Wrap paid commands in the concrete workload timeout chosen from the approved envelope; the workload timeout must expire before the independently controlled provider deadline. A process timeout cannot itself terminate provider billing. Treat skipped/unexecuted CUDA cases as incomplete, never passing validation. The suite checks complex128/complex64 waveform and physical-gradient agreement, Taylor behavior, inversion behavior and device-resident hierarchy data. Complex128 waveform/physical-gradient relative error must be at most `1e-7`; complex64 requires `1e-3` waveform and `1e-2` physical-gradient error plus its Taylor/inversion checks.

`--quick` fixes the solver benchmark at 8 by 12 physical nodes, 5 Hz, two right-hand sides, complex128, deterministic seed and bounded FGMRES iterations/memory; it includes direct CPU comparison, setup, all-RHS timings, implicit gradient, exact HVP, true residual and memory evidence. These are algebraic fixtures and do not validate physical dispersion, PML calibration or full Marmousi reconstruction. Inspect all reported waveform/gradient/HVP errors and solver failures; script exit alone is insufficient evidence. Its normalized log-slowness derivative timings are labeled separately from the physical-gradient precision checks. No performance claim is permitted without matched accuracy and documented hardware/configuration. A failed iterative solve ends the corresponding derivative claim.

Task-12 configuration/manifests remain the source of Marmousi acquisition, axes, units, source coverage, PML/dispersion/resource assumptions. The supplied `run_marmousi_cpu.py` is a CPU runner; do not advertise it as a GPU benchmark. No full Marmousi GPU inversion, hardware sweep or extensive Hessian study fits implicitly into this initial envelope. Any additional small task-12-derived GPU workload must receive explicit numerical/resource preflight and its own command/config record first.

## Transfer, deadline, cleanup and evidence

Before allocation, establish an external controller with a fixed UTC provider shutdown deadline, task-owned resource IDs, periodic billing checks, shutdown retries and emergency termination within the reserve. A delayed command inside the Pod is not independent enforcement. Provider-scheduled shutdown support and mutation permissions have not been verified.

Use an already authorized SSH public key and provider-reported SSH endpoint with verified host key checking. Transfer an archive of the reviewed commit plus only explicitly reviewed configuration files and their hashes. Prefer direct SSH/SCP if the Pod exposes SSH; if only the SSH gateway works, use a tar stream over SSH rather than assuming SFTP support. Never transfer `.env`, private keys or provider credentials. Verify remote Python/PyTorch/CUDA versions, GPU identity, free memory/disk and archive hashes before numerical execution. Startup/transport failure consumes only its bounded allowance; do not open an installation/search loop on the meter.

Preserve small versioned numerical manifests and bounded logs before deleting anything. Documented task-owned compute shutdown/termination routes are `POST https://rest.runpod.io/v1/pods/{podId}/stop` and `DELETE https://rest.runpod.io/v1/pods/{podId}`. Stopping compute can leave billable volume storage. Separately created network volumes require `DELETE https://rest.runpod.io/v1/networkvolumes/{networkVolumeId}` after preserving their required data. These methods are documented capabilities, not confirmed permissions in this session.

Verify provider state through `GET /v1/pods/{podId}` and the list operation, and verify separately created storage absence independently. Remove only task-created billable resources. Re-query attributed billing after metering catches up; report final cost or explicitly unknown final metering. A completed workload or accepted shutdown request alone proves neither stopped billing nor settled cost. Since this preflight created nothing, it has no task-created resource to clean up.

Sources: [Pod pricing](https://docs.runpod.io/pods/pricing), [billing](https://docs.runpod.io/accounts-billing/billing), [Pod lifecycle](https://docs.runpod.io/pods/manage-pods), [REST v1 schema](https://docs.runpod.io/api-reference/openapi.json), [REST v2 schema](https://docs.runpod.io/api-reference-v2/openapi.json).
