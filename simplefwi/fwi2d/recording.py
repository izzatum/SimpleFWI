"""Bounded, lossless HDF5 snapshots of explicitly accepted optimizer states.

CPU arrays/tensors are copied into a two-chunk pool by default. Only the writer
thread touches HDF5. CUDA payloads require a future pinned/event transfer adapter;
this implementation rejects them rather than making an unbounded CPU copy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import threading
from collections import deque
from typing import Any, Mapping

import h5py
import numpy as np

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ArrayPayload:
    """Array with explicit scientific axes and units; source IDs are global IDs."""
    data: Any
    axes: tuple[str, ...]
    units: str
    origin: tuple[float, ...] | None = None
    frequencies: tuple[float, ...] | None = None
    sources: tuple[int, ...] | None = None
    region: str | None = None
    coordinates: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RecordingPolicy:
    path: str | Path
    fields: tuple[str, ...] = ()
    frequencies: tuple[float, ...] | None = None
    sources: tuple[int, ...] | None = None
    cadence: int = 1
    chunk_bytes: int = 8 * 1024 * 1024
    queue_capacity: int = 2
    max_queue_bytes: int | None = None
    compression: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.cadence < 1 or self.chunk_bytes < 1 or self.queue_capacity < 1:
            raise ValueError('cadence, chunk_bytes and queue_capacity must be positive')
        cap = self.chunk_bytes * self.queue_capacity if self.max_queue_bytes is None else self.max_queue_bytes
        if cap < self.chunk_bytes:
            raise ValueError('max_queue_bytes must accommodate one chunk_bytes buffer')
        if self.compression not in (None, 'gzip', 'lzf'):
            raise ValueError('only lossless gzip/lzf compression is supported')
        if any('/' in name or not name for name in self.fields):
            raise ValueError('field names must be nonempty HDF5 basenames')
        object.__setattr__(self, 'max_queue_bytes', cap)
        # Metadata size is separately capped, independent of the payload queue.
        if len(_json(self.metadata).encode()) > 65536:
            raise ValueError('run metadata exceeds 64 KiB')


def _json(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f'unsupported metadata type: {type(item).__name__}')
    return json.dumps(value, default=convert, allow_nan=False)


def _array(value):
    # Duck typing keeps this module usable without importing the torch runtime.
    if hasattr(value, 'detach'):
        if value.device.type != 'cpu':
            raise ValueError('CUDA recording requires an explicit pinned-buffer/event transfer adapter')
        if value.is_conj() or value.is_neg():
            raise ValueError('lazy conjugate/negative tensor views must be resolved by the caller in bounded chunks')
        value = value.detach().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in 'biufc':
        raise ValueError('recording supports numeric arrays only')
    return array


class Recorder:
    """Single-writer context manager with backpressure and bounded copied buffers.

    ``step_result`` structurally supplies ``accepted=True``, ``accepted_step``
    (or ``iteration``) and scalar ``loss``. Rejected trials raise before enqueue.
    Calls from multiple producer threads are serialized. ``close`` always joins
    and raises a writer error; consumers must ignore steps with complete=False.
    An existing archive is never truncated; use a new path after a restart.
    """
    def __init__(self, policy: RecordingPolicy):
        self.policy = policy
        self._condition = threading.Condition()
        self._producer = threading.Lock()
        self._queue = deque()
        self._pool = []
        self._pending_bytes = 0
        self._pending_chunks = 0
        self._buffer_count = 0
        self.peak_buffer_bytes = 0
        self.peak_pending_bytes = 0
        self.peak_pending_chunks = 0
        self._error = None
        self._closing = False
        self._closed = False
        self._last_step = -1
        self._thread = threading.Thread(target=self._worker, name='simplefwi-hdf5-writer', daemon=True)
        self._thread.start()

    @property
    def pending_bytes(self):
        with self._condition:
            return self._pending_bytes

    def _check_error(self):
        if self._error is not None:
            raise RuntimeError('recording writer failed') from self._error

    def _submit(self, item, buffer=None):
        size = 0 if buffer is None else buffer.nbytes
        with self._condition:
            while (self._pending_chunks >= self.policy.queue_capacity or
                   self._pending_bytes + size > self.policy.max_queue_bytes):
                self._check_error()
                if self._closing:
                    raise RuntimeError('recorder is closed')
                self._condition.wait()
            self._check_error()
            if self._closing:
                raise RuntimeError('recorder is closed')
            self._pending_chunks += 1
            self._pending_bytes += size
            self.peak_pending_chunks = max(self.peak_pending_chunks, self._pending_chunks)
            self.peak_pending_bytes = max(self.peak_pending_bytes, self._pending_bytes)
            item['buffer'] = buffer
            self._queue.append(item)
            self._condition.notify_all()

    def _get_buffer(self, count, dtype):
        # Fixed-size buffers have a separate resident-byte cap, including idle
        # pooled buffers and the producer's current copy. A nonintegral byte cap
        # conservatively rounds down the pool capacity to whole chunks.
        size = count * dtype.itemsize
        pool_limit = min(self.policy.queue_capacity,
                         self.policy.max_queue_bytes // self.policy.chunk_bytes)
        with self._condition:
            while True:
                self._check_error()
                if self._closing:
                    raise RuntimeError('recorder is closed')
                room = (self._pending_chunks < self.policy.queue_capacity and
                        self._pending_bytes + size <= self.policy.max_queue_bytes)
                if room and self._pool:
                    raw = self._pool.pop()
                    return raw[:size].view(dtype)
                if room and self._buffer_count < pool_limit:
                    raw = np.empty(self.policy.chunk_bytes, dtype=np.uint8)
                    self._buffer_count += 1
                    self.peak_buffer_bytes = max(self.peak_buffer_bytes,
                                                self._buffer_count * self.policy.chunk_bytes)
                    return raw[:size].view(dtype)
                self._condition.wait()

    def _prepare(self, name, payload):
        if not isinstance(payload, ArrayPayload):
            raise TypeError(f'{name} requires ArrayPayload with axes and units')
        array = _array(payload.data)
        if len(payload.axes) != array.ndim or len(set(payload.axes)) != array.ndim:
            raise ValueError(f'{name}: axes must be unique and match array dimensions')
        if not payload.units:
            raise ValueError(f'{name}: explicit units are required')
        if array.dtype.itemsize > self.policy.chunk_bytes:
            raise ValueError('chunk_bytes is smaller than one array element')
        if 'wavefield' in name and name not in ('physical_wavefield', 'pml_wavefield'):
            raise ValueError('wavefield names must distinguish physical_wavefield/pml_wavefield')
        if 'gradient' in name and name not in ('physical_gradient', 'parameter_gradient'):
            raise ValueError('gradient names must distinguish physical_gradient/parameter_gradient')
        attrs = {'axes': _json(payload.axes), 'units': payload.units, 'dtype': str(array.dtype)}
        indices = [range(n) for n in array.shape]
        for label, coordinate, selected in (
            ('frequencies', payload.frequencies, self.policy.frequencies),
            ('sources', payload.sources, self.policy.sources),
        ):
            axis_names = ('frequency',) if label == 'frequencies' else ('source', 'source_batch')
            axis = next((i for i, key in enumerate(payload.axes) if key in axis_names), None)
            if axis is not None and coordinate is None:
                raise ValueError(f'{name}: {label} coordinates are required for this axis')
            if coordinate is not None:
                if axis is None or len(coordinate) != array.shape[axis]:
                    raise ValueError(f'{name}: {label} coordinate count must match axis')
                # Coordinate vectors are metadata and share its fixed-size cap.
                # Stream the check before making any axis-sized index arrays.
                values = []
                encoded_bytes = 2
                seen = set()
                for value in coordinate:
                    encoded = _json(value)
                    encoded_bytes += len(encoded.encode()) + 1
                    if encoded_bytes > 65536:
                        raise ValueError(f'{name}: {label} coordinates exceed 64 KiB metadata cap')
                    if value in seen:
                        raise ValueError(f'{name}: {label} coordinates must be unique')
                    seen.add(value)
                    values.append(value)
                if selected is None:
                    chosen = range(len(values))
                    output_values = values
                else:
                    wanted = set(selected)
                    chosen = tuple(index for index, value in enumerate(values) if value in wanted)
                    output_values = [values[index] for index in chosen]
                indices[axis] = chosen
                attrs[label] = _json(output_values)
        region = payload.region
        if name in ('physical_wavefield', 'pml_wavefield'):
            expected = 'physical' if name.startswith('physical') else 'pml_padded'
            if region is not None and region != expected:
                raise ValueError(f'{name}: region must be {expected}')
            region = expected
        if region is not None:
            attrs['region'] = region
        if payload.origin is not None:
            attrs['origin'] = _json(payload.origin)
        if payload.coordinates is not None:
            attrs['coordinates'] = _json(payload.coordinates)
        if len(_json(attrs).encode()) > 65536:
            raise ValueError('array metadata exceeds 64 KiB')
        return array, indices, attrs

    def record_accepted(self, step_result, payload: Mapping[str, ArrayPayload]):
        with self._producer:
            with self._condition:
                self._check_error()
                if self._closing:
                    raise RuntimeError('recorder is closed')
            if getattr(step_result, 'accepted', None) is not True:
                raise ValueError('only explicitly accepted steps may be recorded')
            number = getattr(step_result, 'accepted_step', getattr(step_result, 'iteration', None))
            if not isinstance(number, int) or number < 0 or number <= self._last_step:
                raise ValueError('accepted_step must be a nonnegative strictly increasing integer')
            loss = getattr(step_result, 'loss', None)
            if hasattr(loss, 'detach'):
                loss = loss.detach().item()
            loss = float(loss)
            if not math.isfinite(loss):
                raise ValueError('accepted loss must be finite')
            prepared = []
            if number % self.policy.cadence == 0:
                for name in self.policy.fields:
                    if name in payload:
                        prepared.append((name, *self._prepare(name, payload[name])))
            self._submit({'kind': 'begin', 'step': number, 'loss': loss})
            for name, array, indices, attrs in prepared:
                shape = tuple(len(i) for i in indices)
                self._submit({'kind': 'dataset', 'step': number, 'name': name,
                              'shape': shape, 'dtype': array.dtype, 'attrs': attrs})
                total = math.prod(shape)
                elements = self.policy.chunk_bytes // array.dtype.itemsize
                for start in range(0, total, elements):
                    end = min(start + elements, total)
                    buffer = self._get_buffer(end - start, array.dtype)
                    # Advanced indexing selects only this chunk, including for
                    # noncontiguous arrays and source/frequency selections.
                    coords = np.unravel_index(np.arange(start, end), shape)
                    buffer[:] = array[tuple(c if isinstance(index, range) else
                                               np.fromiter((index[int(i)] for i in c), dtype=np.intp, count=len(c))
                                               for index, c in zip(indices, coords))]
                    self._submit({'kind': 'array', 'step': number, 'name': name,
                                  'start': start, 'end': end, 'shape': shape}, buffer)
            self._submit({'kind': 'complete', 'step': number})
            self._last_step = number

    def _write_chunk(self, handle, item):
        """Writer hook; all calls run on the owning worker thread."""
        key = f'steps/{item["step"]:012d}'
        if item['kind'] == 'begin':
            group = handle.create_group(key)
            group.attrs.update(complete=False, accepted_step=item['step'], loss=item['loss'])
        elif item['kind'] == 'dataset':
            ds = handle[key].create_dataset(item['name'], shape=item['shape'], dtype=item['dtype'],
                                             compression=self.policy.compression)
            for attr, value in item['attrs'].items():
                ds.attrs[attr] = value
        elif item['kind'] == 'array':
            ds = handle[f'{key}/{item["name"]}']
            # HDF5 supports hyperslabs, not arbitrary multidimensional point
            # tuples; a chunk crosses at most a bounded set of final-axis rows.
            flat = item['start']
            buffer = item['buffer']
            consumed = 0
            if not item['shape']:
                ds[()] = buffer[0]
                return
            width = item['shape'][-1]
            while flat < item['end']:
                coord = np.unravel_index(flat, item['shape'])
                count = min(width - coord[-1], item['end'] - flat)
                ds[coord[:-1] + (slice(coord[-1], coord[-1] + count),)] = buffer[consumed:consumed + count]
                consumed += count
                flat += count
        else:
            handle.flush()
            handle[key].attrs['complete'] = True
            handle.flush()

    def _worker(self):
        try:
            path = Path(self.policy.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, 'x') as handle:
                handle.attrs['schema_version'] = SCHEMA_VERSION
                handle.attrs['metadata'] = _json(self.policy.metadata)
                handle.attrs['transfer_mode'] = 'cpu_copy'
                handle.create_group('steps')
                while True:
                    with self._condition:
                        while not self._queue and not self._closing:
                            self._condition.wait()
                        if not self._queue:
                            break
                        item = self._queue.popleft()
                    self._write_chunk(handle, item)
                    with self._condition:
                        buffer = item.pop('buffer')
                        if buffer is not None:
                            self._pending_bytes -= buffer.nbytes
                            if len(self._pool) < self.policy.queue_capacity:
                                self._pool.append(buffer.base)
                        self._pending_chunks -= 1
                        self._condition.notify_all()
                handle.flush()
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._queue.clear()
                self._pending_bytes = 0
                self._pending_chunks = 0
                self._condition.notify_all()
        finally:
            with self._condition:
                self._pool.clear()
                self._condition.notify_all()

    def close(self):
        with self._producer:
            with self._condition:
                self._closing = True
                self._condition.notify_all()
            self._thread.join()
            self._closed = True
            self._check_error()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
