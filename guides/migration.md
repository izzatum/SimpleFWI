# Migrating legacy callers

Legacy top-level `simplefwi` exports and entry points remain available in this
release. The new SI implementation is imported from `simplefwi.fwi2d`; it does
not reinterpret old dictionaries or flattened arrays automatically. The legacy
full pytest collection still has pre-existing failures. Targeted new tests are
run with `python -m pytest tests/fwi2d -q`; this is not a claim that the entire
repository test suite passes.

| Legacy input/concept | Explicit new contract |
| --- | --- |
| Geometry dictionary (`n`, `h`, coordinate fields) | `Grid2D(shape=(nz,nx), spacing_m=(dz,dx), origin_m=(z0,x0))` plus `Acquisition` |
| Implicit coordinate axes/units | Coordinates are `(z,x)` in metres, checked against physical node bounds |
| Flattened model | Real `[z,x]` velocity tensor in m/s, float64 by default |
| Slowness-squared array | Physical `m = 1/v²` in s²/m²; obtain from `VelocityParameters.slowness_squared()` |
| Frequency list inside a dictionary | Explicit positive `frequencies_hz` sequence |
| Combined source matrix | One complex amplitude per point source in `Acquisition.amplitudes`; encode source mixing explicitly before migration |
| Implicit data layout | Complex `[frequency,source,receiver]` with documented source IDs and frequency coordinates |
| Legacy optimizer routine | Native `torch.optim` instance or factory passed to `InversionRunner` |
| Backend selector | Explicit `SolverConfig(method,device,dtype)`; CPU direct is the reference default |
| Iteration dump | Accepted-state HDF5 archive plus separate validated atomic restart checkpoint |

Determine original array ordering before reshaping. For a legacy NumPy array
flattened from `[z,x]` with Fortran ordering, for example:

```python
import numpy as np
import torch

# legacy_m, nz and nx come from the original caller's documented layout.
m_si = np.asarray(legacy_m).reshape((nz, nx), order='F')
# If legacy_m is measured in s²/km², convert before taking the square root.
m_si = m_si / 1_000_000.
velocity = torch.tensor(1. / np.sqrt(m_si), dtype=torch.float64)
```

Omit that conversion only when the input is already s²/m². A velocity of
2 km/s becomes 2000 m/s, with physical m = 2.5e-7 s²/m². Do not assume historical
examples all used the same normalization. Convert coordinates and spacing
independently, and preserve the physical origin. NumPy C ordering may be the
correct choice for another caller; guessing a reshape order can transpose the
physical model without a shape error.

Recompute observations when migrating unless the old source normalization,
Fourier sign, discrete operator and data axes have been established. The new
point-source RHS divides amplitudes by cell area; outputs use normalized
`source_unit`, not an inferred pressure calibration. Frequency/source axes may
need explicit transposition, not just reshaping. There is no automatic old/new
numerical equivalence guarantee.

Start with the [runnable README](../README.md), then compare a homogeneous
forward solve and independent finite-difference derivative checks. Validate the
new grid/PML/frequency combination using [accuracy guidance](accuracy.md).
Preserve dataset/model/config hashes, optimizer settings and units with the run;
legacy iteration files lack enough state to become restart checkpoints without
an explicit, verified conversion. See [architecture](architecture.md) and
[output/restart](outputs.md).
