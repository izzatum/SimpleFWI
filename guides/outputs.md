# Scientific output and restart

`simplefwi.fwi2d.recording` writes lossless local HDF5 archives. A single writer
thread owns the file; `record_accepted` copies numeric CPU arrays or detached
PyTorch tensors into bounded reusable buffers. Always use a context manager or
call `close()`: it flushes, joins the writer, and propagates disk errors. An
existing archive is rejected rather than overwritten. Use a fresh archive path
when continuing a run.

```python
from types import SimpleNamespace
import numpy as np
from simplefwi.fwi2d.recording import ArrayPayload, Recorder, RecordingPolicy

policy = RecordingPolicy(
    "run-001.h5", fields=("model",), chunk_bytes=1024 * 1024,
    metadata={"source_normalization": "cell_area", "seed": 7,
              "solver_tolerances": {"rtol": 1e-9}},
)
with Recorder(policy) as recorder:
    recorder.record_accepted(
        SimpleNamespace(accepted=True, accepted_step=1, loss=0.25),
        {"model": ArrayPayload(np.full((8, 12), 1500.), ("z", "x"),
                               "m/s", origin=(0., 0.), region="physical")},
    )
```

The structural step interface is `accepted=True`, a strictly increasing integer
`accepted_step` (`iteration` is also accepted), and finite scalar `loss`.
Rejected trial closures must not invoke accepted-state callbacks: an explicit
`accepted=False` is rejected before writing. This interface cannot establish
optimizer acceptance from tensor values; the optimizer driver owns that decision.
A diagnostic group is written for every accepted call, including steps excluded
by snapshot cadence. Arrays are opt-in (`fields=()` by default); wavefields and
gradients are off. Model snapshots can be selected with `fields=("model",)` at
`cadence=1`, matching the recommended every-accepted-step checkpoint cadence.
Neither the recorder nor the checkpoint primitive automatically schedules
optimizer callbacks.

## Axes and provenance

Every array requires unique axis names matching its dimensions and explicit
units. Use `[z, x]` for physical grids, `[frequency, source, receiver]` for data,
and `[source_batch, z_padded, x_padded]` for padded batched wavefields. Supply SI
origins and coordinate spacing/positions in `origin` and `coordinates`; these are
saved without changing coordinates. Spatial origins are application inputs and
must refer to the actual physical or padded array. The recorder does not infer
an omitted origin. Use `physical_wavefield` / `pml_wavefield` and
`physical_gradient` / `parameter_gradient` to distinguish regions and derivative
coordinates. `pml_wavefield` describes the entire PML-padded array, not only its
outer strip. Gradient units depend on objective and parameterization and must be
supplied explicitly.

Axes, units, dtype, region, origin, coordinates, frequency values and global
source IDs are dataset attributes (structured values use JSON). Frequency/source
axes require matching coordinate vectors; duplicate IDs are rejected. Policy
`frequencies` and `sources` select coordinate values rather than positions.
Coordinate vectors are also limited to 64 KiB each before selection, so batched
inputs with oversized source/frequency metadata fail before allocation. Selections
intersect each supplied batch, preserve its axis ordering, and can
produce a zero-length axis if no requested source belongs to that batch. Supply
selected batch arrays, rather than retaining a full differentiable survey.

Use run `metadata` for physical model identity, package versions, Git revision,
seeds, source normalization, precision, solver tolerances and Fourier convention
`exp(-i omega t)`. The recorder preserves supplied metadata; it does not invent
missing provenance. Each step lives at `/steps/{accepted_step:012d}`. Read only
groups with `complete=True`; a writer failure can leave an incomplete group. HDF5
is an archive, not a crash-atomic restart format: a process kill can damage an
open archive. Restart state uses the separate atomic checkpoint primitive.

## Memory, transfers and errors

`chunk_bytes` defaults to 8 MiB and `queue_capacity` defaults to two. The default
`max_queue_bytes` is their product. Queued **and actively written** copied array
bytes count against the queue cap. A producer waits when either limit is reached.
Reusable fixed-size buffers, including idle pooled buffers and the current
producer buffer, are also bounded by `max_queue_bytes`. A cap that is not a
multiple of `chunk_bytes` conservatively rounds down the resident pool capacity.
`peak_pending_bytes`, `peak_pending_chunks`, and `peak_buffer_bytes` expose these
bounds. Run/dataset metadata has a separate 64 KiB per-object cap. HDF5 caches,
Python metadata, input arrays and temporary index/gather arrays are outside the
copied-buffer cap; gather scratch is bounded by one output chunk and its rank.
Unselected axes use lazy ranges, so preparing a large grid does not allocate full
axis-index arrays. Source/frequency coordinate validation and selection scratch
is bounded by the separate 64 KiB input-coordinate metadata limit.
This is a bound on recording working storage, not on complete process RSS or
mandatory optimizer state.

Compression defaults to `None`; optional `gzip` and `lzf` are lossless. Tensor
payloads are detached and copied before `record_accepted` returns, so subsequent
mutation cannot alter the snapshot and the queue holds no autograd graph. CPU
recording is implemented and tested for tensors without unresolved conjugate or
negative view bits. Lazy conjugate/negative views are explicitly rejected before
copying; callers can resolve independently bounded slices before recording.
CUDA tensors raise an explicit error: a
pinned-buffer/event transfer path is not implemented or validated here. Copying a
whole CUDA tensor to CPU before calling the recorder is the caller's separate
memory allocation and transfer cost, not bounded GPU recording.

## Checkpoints

`capture_checkpoint(model, optimizer, scheduler=None, *, accepted_step, config,
metadata=None)` clones an accepted state to CPU. `model` supplies native
`state_dict` / `load_state_dict`. The state includes the native optimizer and
scheduler dictionaries/classes, model shapes and dtypes, optimizer parameter
group shapes, accepted counter, Python/NumPy/Torch RNGs, already-initialized CUDA
RNGs, configuration and metadata. Metadata automatically includes Torch and NumPy
versions. Supply seed/Git revision and other run provenance explicitly.

`config` must contain nonempty `parameterization`, `dataset_hash`, `model_hash`,
and `config_hash`. Applications compute these from canonical dataset/model/config
content before saving; the checkpoint API compares the supplied identities and
does not independently hash an external dataset. `config_hash` should cover
solver tolerances, source normalization, geometry, optimizer options including
weight decay, regularization, and all other restart-sensitive choices.

```python
from simplefwi.fwi2d.checkpoint import (
    capture_checkpoint, save_checkpoint, load_checkpoint, restore_checkpoint,
)

# model, optimizer, scheduler are the current native PyTorch objects.
# Identity strings below must be computed from the real run inputs.
state = capture_checkpoint(model, optimizer, scheduler, accepted_step=step,
                           config=run_config, metadata={"seed": 7})
save_checkpoint("accepted.pt", state)
loaded = load_checkpoint("accepted.pt", expected_config=run_config)
step = restore_checkpoint(loaded, model, optimizer, scheduler)
```

`save_checkpoint(path, state)` writes a unique temporary file in the same directory,
flushes and fsyncs it, verifies the staged file using `weights_only=True`,
atomically replaces the target, and fsyncs the directory. Before staging, the
entire state is normalized to plain built-in containers/scalars and native
tensors; NumPy scalar/array provenance becomes built-in values/lists, and
unsupported custom leaves are rejected. Expected config uses the same
normalization. Staged readback temporarily allocates another complete checkpoint
state; it is separate from the archival recorder buffer budget.
A serialization failure or failure before rename preserves the previous complete
checkpoint and removes the temporary file. The single atomically published file
is the commit record; no independently updated manifest can disagree with it.
This relies on the supported local filesystem's atomic rename/fsync semantics.
A post-rename durability failure can raise even if the new complete file exists.

`load_checkpoint(path, expected_config)` loads tensors on CPU with Torch's
`weights_only=True`, validates schema and all four identity fields, and checks
additional expected config values recursively. It can also check
`optimizer_class`, `scheduler_class` and `model_shapes`. `restore_checkpoint`
checks model shapes/precision, native optimizer class/group shapes, and scheduler
class before applying state. RNG restoration is on by default; CUDA RNG device
count must match. Passing `restore_rng=False` explicitly omits all RNG restoration
and therefore forfeits stochastic restart equivalence. Exact deterministic CPU
Adam + StepLR continuation is tested, including optimizer moments and random
number streams. This does not establish equivalence across Torch releases,
devices or nondeterministic kernels.

## Measured local cost

A Python 3.12 / Torch 2.12.0+cpu microbenchmark used sixteen accepted snapshots,
each with four complex128 sources on a 96 by 96 padded grid, uncompressed output,
256 KiB chunks, and two buffers. Both runs performed identical array updates and
inner products (`OPENBLAS_NUM_THREADS=1`). The separate processes reported:

| Mode | Timed loop + close | Peak host RSS | Copied buffer peak | Output |
| --- | ---: | ---: | ---: | ---: |
| off | 0.000578 s | 341076 KiB | 0 | 0 |
| on | 0.543018 s | 345716 KiB | 524288 bytes | 9478512 bytes |

The payload was 9437184 bytes (about 16.6 MiB/s over the timed on run). Initial RSS
was 341076 / 340980 KiB respectively; the on process's increase was 4736 KiB.
These are single local measurements including asynchronous flush/join, not an
FWI solve benchmark or speedup claim. HDF5/allocator caches contribute to RSS.
No CUDA device was used; peak device memory was **not measured**. The complete
microbenchmark recipe and focused test evidence are recorded in
`.superpowers/sdd/fwi-redesign/task-9-report.md`.
