# Bounded RunPod validation protocol and attempted execution

**GPU numerical validation remains incomplete. Both earlier task-created Pods were terminated and provider absence verified.** Two bounded allocations of one A40 at a time were attempted, with no numerical GPU workload executed. [Structured lifecycle evidence](results/runpod-attempts.json) records the exact timeline, budget and cleanup; [preflight status](results/runpod-preflight.json) preserves the access history.

## Latest retry: 15 October 2026, 15:35 UTC

The user-requested [read-only retry](results/runpod-retry.json) was blocked before allocation. The newly published `RUNPOD_API_KEY` binding permits only `rest.runpod.io`; its previously working `api.runpod.io` destination is missing. REST inventory and billing still return Cloudflare HTTP 403/error 1010, while API account and billing return HTTP 401. This does not establish that the real key expired. No new resource was created, no new allocation charge was incurred, and no GPU workload ran. Current inventory and settled billing could not be refreshed; the earlier deletion confirmations below remain historical evidence.

Edit the **existing** secret in environment settings to allow both `api.runpod.io` and `rest.runpod.io`, then save and publish. The configuration tool rejected this destination change with `INVALID_ARGUMENT` and explicitly requires editing the existing entry; no binding change was saved and no duplicate secret is needed. Restoring the binding permits a fresh control-plane check; it does not by itself resolve the earlier SSH/Jupyter access failures. Recheck control, billing and a supported workload path before another bounded allocation.

## Observed outcome on 15 October 2026

| Phase | Result |
| --- | --- |
| Initial pre-publication checks | Historical account/pricing HTTP 401 and REST HTTP 403; not proof the real key expired |
| Fresh published-runtime account checks | GraphQL/API-v2 HTTP 200; credit sufficient for the USD 10 budget; current rate USD 0/hour; no existing compute/storage resources; today's itemized history reported zero |
| Actual A40 quote/allocation | One NVIDIA A40 at USD 0.59/hour, 40 GB ephemeral disk, no persistent/network/global mounts |
| Attempt 1, SSH transport | Pod `144hxuu5czqg3h`, exposed 22/tcp; provider-reported direct SSH connection refused and supported SSH gateway handshake unavailable |
| Initial HTTP probe during attempt 1 | Port 19123 was not explicitly exposed; unauthenticated HTTP 403 with body not retained. This probe was inconclusive for HTTP execution |
| Local corrected HTTP client proof | Real loopback Jupyter contents upload/download, kernel WebSocket execution, SHA-256 verification and safe extraction passed; ephemeral authentication token not logged |
| Attempt 2, explicitly configured Jupyter | Pod `qzu87v7wplhu0d`, exposed 8888/http, `startJupyter=true`, fresh workload-token header; authenticated contents request returned HTTP 403 with body `error code: 1010` |
| Final provider check, 23:16:11 MYT (UTC+8) | Both individual Pod GETs returned 404; complete Pod/network-volume/global-volume/serverless/cluster inventories empty; account rate USD 0/hour |
| Accounting snapshot | Credit sufficient for the task budget; observed account debit USD 0.0623020102; final itemized task cost unknown because history still lagged |

Attempt 1 was created at 22:59:09 MYT and provider-confirmed absent at 23:05:33 MYT. Attempt 2 was created at 23:13:29 MYT and provider-confirmed absent at 23:14:22 MYT. These times are Malaysia Time (UTC+8); structured audit records use UTC. Each controller issued DELETE, confirmed GET 404 and two successive task-name-filtered empty inventory checks. No task-created billable resource remains, no persistent storage was created, and no unrelated resource was changed. There was no third allocation.

The corrected 8888/http denial was a provider Cloudflare error 1010 response despite the supplied Jupyter token, rather than evidence of a missing API key or an expired real key. It prevented reaching a usable Jupyter session. No User-Agent/WAF/proxy bypass was attempted. A supported workload connection permitted by the provider/site-owner access policy is needed before another paid experiment. The later missing credential destination is a separate issue, described above; a duplicate API secret is not needed.

The pinned image was `runpod/pytorch:1.4.0-cu1300-torch2120-ubuntu2404` at digest `sha256:93396b43c615b051e3951adbed2f958acff7d645936caa4029a58d0c1d309abd`. Its tag/digest and provider-reported A40 allocation were verified. **Python, PyTorch, CUDA and driver versions inside the GPU container were not observed by remote code.** No GPU accuracy, timing, memory or speedup result is claimed. The existing CPU manifests retain their original zero RunPod costs because those CPU runs were separate.

## CPU, accounting and budget gates

The root task supplied the committed CPU correctness gate at revision `9e415d1c66a922cce2b6d2b5151df38e7e8dd40a`: 367 passing tests and 8 CUDA cases explicitly unrun. Re-run the relevant gate after any numerical change:

```bash
python -m pytest tests/fwi2d -m 'not cuda' -q
```

Total expenditure for this entire task must remain **at most USD 10**, including every allocation attempt, compute, setup/download time, disks, retained storage, shutdown latency and metering lag. Reserve **at least USD 2** for overhead/shutdown/storage. Compute allocation must be **at most USD 8 minus already incurred task costs**, reduced further if the reserve is insufficient. This budget does not reset for a retry.

For each attempt, the external controller was armed before creation with a 45-minute maximum wall deadline, a 10-minute startup/setup cutoff, a USD 0.75/hour provider-compute-rate guard and a USD 1 cumulative account-spend guard. The current actual compute quote was USD 0.59/hour. Forty GB of ephemeral disk at the documented USD 0.10/GB/month was bounded conservatively by USD 4/672 hours. A 45-minute compute-plus-disk plan was about USD 0.447; the conservative rate guard plus USD 2 reserve bounded an attempt's envelope at USD 2.567. Attempt 2 additionally reserved USD 0.10 for attempt 1, for USD 2.667 total, below the USD 10 ceiling.

The two actual allocated-to-confirmed-absence durations give a compute-plus-disk estimate of about USD 0.07228. This is an estimate, not a settled bill. Provider credit observations show a USD 0.0623020102 debit; itemized history still returned zero records after more than five minutes and cannot be interpreted as free usage. Keep final settled cost unknown until provider metering resolves.

In general, require:

```text
confirmed_prior_cost + bounded_fixed_overhead + all_resource_hourly_rate * wall_hours + reserve <= USD 10
reserve >= USD 2
compute_allowance <= USD 8 - confirmed_prior_cost
```

If any term cannot be bounded, do not launch. An account hourly spending limit does not enforce this task's total-dollar ceiling. Metering lag and billing units need explicit allowances. No automatic budget increase, hardware sweep, full Marmousi inversion or extensive Hessian study is covered.

## Supported control-plane and workload transport

GraphQL at `https://api.runpod.io/graphql` and documented API-v2 routes were usable. Read-only routes include `/v2/pods`, `/v2/network-volumes`, `/v2/global-volumes`, `/v2/serverless`, `/v2/clusters`, `/v2/catalog/gpus` and `/v2/billing`; resource-specific Pod billing is `/v2/billing/pods`. Use explicit UTC billing windows and resource IDs to attribute spending. REST-v1 Cloudflare denials are historical and are not required for control when APIv2 works.

APIv2 publishes Pod lifecycle/logging routes, but no command-execution endpoint. Normal SSH requires an actually reachable provider SSH endpoint and verified host key. The first attempt obtained the container host key through authenticated logs, yet the reported direct endpoint refused connections.

The corrected supported HTTP mechanism explicitly exposes 8888/http and enables Jupyter on an official compatible image. `JUPYTER_PASSWORD` is one fresh workload authentication token; it is independent of the provider API credential. Supply it only through the authorization header, never a URL. Use documented Jupyter `PUT /api/contents/{path}` for an encoded archive, `POST /api/kernels`, the kernel channels WebSocket, `GET /api/contents/{path}` to retrieve evidence and `DELETE /api/kernels/{id}`. The client used system CA roots and hostname verification for HTTPS/WSS. A real local server proved upload, execution, archive hashing/extraction, download and kernel cleanup; the cloud 8888 endpoint was denied before an actual kernel WebSocket could be attempted.

The transport proof is separate from GPU validation. Keep provider credentials outside the Pod. Never print request headers, token-bearing URLs, environment dumps, private keys or credential values. Temporary private SSH/workload-token files remain local and are removed after their resources are gone; they are absent from public artifacts.

## Exact pending numerical workloads

These commands were prepared but **not executed on the GPU**. A future permitted workload connection still requires a fresh account/quote/budget check and external controller before allocation:

```bash
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"; assert torch.cuda.device_count() == 1, "one GPU required"; print(torch.__version__, torch.cuda.get_device_name(0))'
python -m pytest tests/fwi2d -m cuda -q -rs --junitxml=task13-results/cuda-tests.xml
python benchmarks/fwi2d/solver.py --device cpu --quick --maxiter 120 --output task13-results/solver-cpu-complex128.json
python benchmarks/fwi2d/solver.py --device cuda --quick --maxiter 120 --output task13-results/solver-cuda-complex128.json
```

Run the CPU comparison before allocation when feasible. Reject any missing/skipped CUDA case; 8 cases were expected from the reviewed revision. The prepared check required zero failures/errors/skips. Complex128 waveform/physical-gradient agreement requires relative error at most 1e-7; complex64 requires waveform at most 1e-3 and physical gradient at most 1e-2 plus its Taylor/inversion gates. Preserve all true residual and float64/complex128 checks; never fall back silently to CPU or lower precision.

`--quick` selects the fixed 8 by 12 physical-node, 5 Hz, two-RHS algebraic case with bounded iterations/memory, direct CPU comparison, setup/all-RHS/gradient/exact-HVP timing and residual/memory evidence. Inspect every solve status and accuracy metric; a successful script exit alone is insufficient. The prepared complex128 benchmark check required true relative residual at most 1e-10, waveform/gradient error at most 1e-7 and HVP error at most 1e-5. These are algebraic checks, not full heterogeneous dispersion/PML or inversion validation.

The reviewed production archive had SHA-256 `c81e13e4ac7c80085b1b0c90c7402ee165ee6e454d918155c147adf4705db084`. Tutorial 01 was separately snapshotted from commit `27142a49ae35464debdf202e4d0e79e7c98669fa`, SHA-256 `d14d232abfd3b7871de190b22c5a7be1b3c0ff155e5df8c34c923994d6c0be1b`; its existing CPU execution is separate from its unrun GPU branch. Full-model task 12 CPU benchmarks are not GPU benchmarks. Preserve dataset/config hashes, SI units/axes, source coverage, solver/PML/dispersion limits and independent-accuracy limitations in any future task 12-derived case.

## Cleanup and evidence

The controller ran outside the Pod in a detached local session, monitored account billing and only the exact prechecked task resource name, and terminated immediately after each transport failure. It used `DELETE https://api.runpod.io/v2/pods/{id}` rather than merely stopping compute. Its ownership guard was tested locally with an unrelated mocked resource and never selected that unrelated ID. A controller heartbeat was verified before each allocation. Native provider-scheduled deadlines were not exposed by the inspected schema; the external controller enforced the deadline.

A finished Python process does not stop billing. Preserve required result manifests before termination, verify provider 404 and list absence, and verify separately created volumes if any. Here no numerical results existed to retrieve and no persistent volume needed deletion. Final inventory confirmed both task Pods absent and all listed resource classes empty. Saved lifecycle events and account snapshots distinguish confirmed deletion, estimated allocation charges, observed credit debits and unresolved final metering.

Sources: [RunPod connections](https://docs.runpod.io/pods/connect-to-a-pod), [official container startup contract](https://github.com/runpod/containers/blob/main/container-template/start.sh), [pricing](https://docs.runpod.io/pods/pricing), [billing](https://docs.runpod.io/accounts-billing/billing), [REST APIv2 schema](https://docs.runpod.io/api-reference-v2/openapi.json).
