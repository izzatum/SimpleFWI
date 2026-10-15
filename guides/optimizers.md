# Native PyTorch optimizers

This implementation targets Python 3.12 and released PyTorch **2.12.0**. The
compatibility inventory discovers native `torch.optim.Optimizer` subclasses and
fails when an exported class has no executable recipe. All 15 classes below
passed CPU float64 parameter/gradient/state checks. These are compatibility
checks; they do not imply universal convergence or support for every native
flag on every device.

```python
import torch
from simplefwi.fwi2d.parameters import VelocityParameters
from simplefwi.fwi2d.optimizers import make_optimizer
from simplefwi.fwi2d.inversion import InversionRunner

# objective is an FWIObjective with fixed acquisition/frequencies/weights.
# initial_velocity has physical [z, x] shape and SI units (m/s).
model = VelocityParameters(initial_velocity, (1400., 2200.), active_mask)
optimizer = make_optimizer('Adam', model, lr=0.05)
runner = InversionRunner(objective, model, optimizer, source_batch_size=2)
result = runner.run(10)
velocity = result.velocity  # detached accepted model in m/s
```

Caller-created native instances and factories also work:
`InversionRunner(objective, model, lambda p: torch.optim.Adam(p, lr=0.05))`.
Native construction arguments pass through unchanged. The runner checks the
optimizer owns exactly the model parameters.

| Native class | Representation | Executed recipe arguments |
| --- | --- | --- |
| Adafactor | dense 2D | `lr=0.01` |
| Adadelta | dense 2D | `lr=1.0` |
| Adagrad | dense 2D | `lr=0.01` |
| Adam | dense 2D | `lr=0.01` |
| Adamax | dense 2D | `lr=0.01` |
| AdamW | dense 2D | `lr=0.01, weight_decay=0` |
| ASGD | dense 2D | `lr=0.01, lambd=0` |
| LBFGS | dense 2D | `lr=1, max_iter=5, history_size=5, line_search_fn='strong_wolfe'` |
| Muon | dense 2D | `lr=0.01, weight_decay=0` |
| NAdam | dense 2D | `lr=0.01` |
| RAdam | dense 2D | `lr=0.01` |
| RMSprop | dense 2D | `lr=0.01` |
| Rprop | dense 2D | `lr=0.01` |
| SGD | dense 2D | `lr=0.01` |
| SparseAdam | sparse embedding | `lr=0.01` |

Unspecified optional flags use native defaults. Muon requires a matrix, which
is why the default `theta` parameter retains [z,x] shape. CPU float64 default
routes are tested; fused, foreach, CUDA and mixed-precision variants are not
certified by this table. `capturable=True` and `differentiable=True` steps are
explicitly rejected: this runner uses Python solver/control flow and detached
batch gradient accumulation.

## Physical mapping and sparse route

For active cells the model maps `theta` to
`v_min + (v_max-v_min)*sigmoid(theta)` and returns `m=1/v**2` in s²/m².
Initial active velocities must be strictly inside finite positive bounds.
Frozen cells are immutable copied background values, independent of optimizer
weight decay. Diagnostics report physical extrema and active saturation counts
(sigmoid fractions outside `[1e-6, 1-1e-6]`). Strong saturation can slow inversion.
The default parameter dtype is float64 for Python arrays; explicit floating
PyTorch inputs retain their dtype/device.

`representation='sparse_embedding'` uses one embedding row per active cell and
produces sparse gradients for native SparseAdam. It evaluates the identical
physical objective and does not imply sparse PDE computation: every active
cell can receive a gradient. Dense parameters with SparseAdam are rejected
before any solve. The sparse runner route currently supports SparseAdam only;
other sparse/native combinations are rejected explicitly.

`model.raw_parameter` returns the dense theta or embedding weight. Dense
`model.slowness_squared_from(theta)` is a pure differentiable transform with
no parameter mutation. Exact theta Hessians must differentiate this transform
twice, retaining its second derivative. Sparse models use physical-coordinate
Hessian actions; sparse native higher-order autodiff is not promised.

The physical regularizer is evaluated in m once per closure, independently of
native weight decay. Native decay values and optimizer class are reported in
accepted diagnostics. Weight decay on theta is a different prior from spatial
regularization in physical m.

## Closure, accepted steps and recording

`source_batch_size` defaults to one and must be positive. Each closure clears
parameter gradients, evaluates each data-only batch at a detached physical
leaf, frees that batch graph immediately, then adds one regularizer. A single
chain-rule pullback maps the accumulated physical gradient into dense or
sparse native parameters. This keeps one physical gradient and one batch graph;
it does not retain all differentiable wavefields. `runner.closure()` returns a
detached scalar. `last_physical_gradient` is the most recent **closure** physical
gradient, which may belong to an LBFGS trial rather than the accepted model.

The source set, frequencies, weights, grid, stencil, PML, and stage must stay
fixed within a step. LBFGS can evaluate its closure many times; the returned
`StepResult` counts closure evaluations separately from `accepted_step`.
An accepted result always recomputes the streamed loss at the final parameters;
native LBFGS's returned original loss is not used. Solver exceptions and
nonfinite loss, gradient, parameter, or optimizer state propagate and restore
the pre-step model/optimizer state. No accepted event is emitted for such a
failure. Rollback keeps an additional copy of parameter/native optimizer state;
LBFGS history remains a parameter-count-dependent memory cost.

`run(steps)` returns scalar diagnostic history plus detached final velocity and
slowness squared. It stores no autograd graphs or per-step model arrays.
A supplied recorder receives exactly one `record_accepted(result,payload)` per
accepted outer step. The payload contains `velocity` and `slowness_squared`
`ArrayPayload` values with physical [z,x] axes, SI units and grid origin where
available. Closure-trial gradients are never recorded as accepted gradients.
Recording cadence/field selection is controlled by the recording policy.

Native schedulers run after accepted states: ordinary schedulers receive
`step()`, and `ReduceLROnPlateau` receives `step(accepted_loss)`. Scheduler and
writer failures propagate. A restart must restore model/optimizer/scheduler
state, RNGs and `runner.accepted_steps`; see [output and restart guidance](outputs.md).
Use `capture_checkpoint(..., accepted_step=result.accepted_step, ...)` after
a successful outer step, never inside its closure.

## Measured PDE integration

The checked-in integration test uses the pinned 48×48 Camembert physical grid,
10 m spacing, 80-cell PML, 5 Hz, eight sources, 32 receivers and a central 16×16
active window. CPU float64/complex128 sparse-direct solves use batches of two.
The objective has unnormalized unit data weights and no regularizer. Same-grid
synthetic observations test optimizer/PDE integration; these results do not
establish reconstruction accuracy against independent data.

Starting loss was `0.0202130395532`. Adam (`lr=0.2`, six steps) reached
`0.00318802696264`; LBFGS (`lr=1`, three outer steps, `max_iter=5`, strong Wolfe)
reached `3.29519312475e-9`, using 6, 6 and 7 closure evaluations. These selected
runs establish reduction, without requiring monotonic Adam steps or claiming
superiority across objectives. Native inventory fixtures also verify bounds,
frozen cells, sparse gradients, and finite nested optimizer state.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/fwi2d/test_parameters.py tests/fwi2d/test_optimizer_inventory.py \
  tests/fwi2d/test_closure.py tests/fwi2d/test_inversion.py -q
```

## Accepted-state recording and checkpoint integration

Attach the public `AcceptedStateRecorder` through
`InversionRunner(..., accepted_callbacks=(adapter,))` to bind archival output and
restart checkpoints to accepted outer steps. Construct the adapter with an open
`Recorder` and a `config` containing `parameterization`, `dataset_hash`,
`model_hash`, and `config_hash`. The adapter captures native model, optimizer,
scheduler and RNG state after the accepted scheduler update. Its
`checkpoint_cadence=1` default writes an atomic checkpoint every accepted step,
independently of the HDF5 snapshot cadence. Without an explicit checkpoint path,
it writes beside the archive as `<archive-path>.checkpoint.pt`.

`RecordingPolicy(fields=(), cadence=1)` defaults to lightweight accepted loss,
closure counts and diagnostics without model arrays, gradients or wavefields.
Accepted diagnostics persist even on steps omitted by array snapshot cadence.
Select `velocity` and/or `slowness_squared` for model snapshots. Selecting
`physical_gradient` makes the adapter recompute the streamed physical gradient at
the accepted model on snapshot steps, including the physical regularizer. It
does not reuse an LBFGS closure-trial gradient. Wavefields remain opt-in:
`AcousticFWI.simulate(..., recorder=...)` can write selected source/frequency
fields to a separate simulation archive.

Use the adapter as the archive's accepted-step producer; avoid passing the same
recorder both as `runner.recorder` and through the callback. A bare
`runner.recorder` provides accepted model payloads and diagnostics but does not
schedule restart checkpoints or recompute accepted gradients. Recording errors
propagate after an optimizer state may already have been accepted. Close the
recorder context to observe asynchronous writer failures. See
[output and restart guidance](outputs.md) for field metadata, buffer limits,
explicit checkpoint examples and restart identity requirements.
