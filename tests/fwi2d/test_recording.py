"""Scientific recording contracts: accepted state, bounded copies and writer ownership."""
from types import SimpleNamespace
import gc
import json
import threading
import time
import weakref

import h5py
import numpy as np
import pytest

from simplefwi.fwi2d.recording import ArrayPayload, Recorder, RecordingPolicy


def accepted(n=1, **kwargs):
    return SimpleNamespace(accepted=True, accepted_step=n, loss=2.5 / n, **kwargs)


def test_lossless_complex_roundtrip_with_axes_and_source_selection(tmp_path):
    path = tmp_path / 'recording.h5'
    value = np.arange(3 * 2 * 4).reshape(3, 2, 4).astype(np.complex128) * (1 + 2j)
    policy = RecordingPolicy(path, fields=('physical_wavefield',), sources=(7,),
                             chunk_bytes=32, metadata={'source_normalization': 'cell_area'})
    with Recorder(policy) as recorder:
        recorder.record_accepted(accepted(), {
            'physical_wavefield': ArrayPayload(value, ('source_batch', 'z', 'x'), 'Pa',
                origin=(10., 20.), sources=(5, 7, 9), region='physical')})
    with h5py.File(path, 'r') as f:
        step = f['steps/000000000001']
        assert step.attrs['complete']
        np.testing.assert_array_equal(step['physical_wavefield'][:], value[1:2])
        ds = step['physical_wavefield']
        assert json.loads(ds.attrs['axes']) == ['source_batch', 'z', 'x']
        assert json.loads(ds.attrs['sources']) == [7]
        assert json.loads(ds.attrs['origin']) == [10., 20.]
        assert ds.attrs['units'] == 'Pa'
        assert ds.attrs['region'] == 'physical'
        assert ds.attrs['dtype'] == 'complex128'
        assert json.loads(f.attrs['metadata'])['source_normalization'] == 'cell_area'


def test_frequency_source_selection_and_cadence_keep_all_diagnostics(tmp_path):
    path = tmp_path / 'data.h5'
    values = np.arange(2 * 3 * 4.).reshape(2, 3, 4)
    with Recorder(RecordingPolicy(path, fields=('predicted_data',), frequencies=(4.,),
                                  sources=(12,), cadence=2)) as recorder:
        for n in (1, 2):
            recorder.record_accepted(accepted(n), {'predicted_data': ArrayPayload(
                values, ('frequency', 'source', 'receiver'), 'Pa',
                frequencies=(2., 4.), sources=(10, 11, 12))})
    with h5py.File(path, 'r') as f:
        assert len(f['steps']) == 2
        assert 'predicted_data' not in f['steps/000000000001']
        np.testing.assert_array_equal(f['steps/000000000002/predicted_data'][:], values[1:2, 2:3])
        assert json.loads(f['steps/000000000002/predicted_data'].attrs['frequencies']) == [4.]


def test_rejected_closures_and_duplicate_accepted_steps_are_not_archived(tmp_path):
    path = tmp_path / 'data.h5'
    with Recorder(RecordingPolicy(path)) as recorder:
        with pytest.raises(ValueError, match='accepted'):
            recorder.record_accepted(SimpleNamespace(accepted=False, accepted_step=1, loss=3.), {})
        recorder.record_accepted(accepted(), {})
        with pytest.raises(ValueError, match='increas'):
            recorder.record_accepted(accepted(), {})
    with h5py.File(path, 'r') as f:
        assert len(f['steps']) == 1


def test_default_wavefields_and_gradients_off(tmp_path):
    path = tmp_path / 'data.h5'
    with Recorder(RecordingPolicy(path)) as recorder:
        recorder.record_accepted(accepted(), {
            'physical_wavefield': ArrayPayload(np.ones((2, 2)), ('z', 'x'), 'Pa'),
            'physical_gradient': ArrayPayload(np.ones((2, 2)), ('z', 'x'), 's/m')})
    with h5py.File(path, 'r') as f:
        assert list(f['steps/000000000001']) == []


def test_two_chunk_byte_bound_and_backpressure(tmp_path, monkeypatch):
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    original = Recorder._write_chunk
    def slow(self, handle, item):
        if item['kind'] == 'array':
            entered.set()
            assert release.wait(5), 'test writer was never released'
        return original(self, handle, item)
    monkeypatch.setattr(Recorder, '_write_chunk', slow)
    recorder = Recorder(RecordingPolicy(tmp_path / 'data.h5', fields=('model',),
                         chunk_bytes=32, max_queue_bytes=64))
    failure = []
    def produce():
        try:
            recorder.record_accepted(accepted(), {'model': ArrayPayload(
                np.arange(100.), ('x',), 'm/s')})
        except BaseException as exc:
            failure.append(exc)
        finally:
            done.set()
    producer = threading.Thread(target=produce)
    producer.start()
    assert entered.wait(5)
    assert not done.wait(.05)
    assert recorder.peak_pending_chunks <= 2
    assert recorder.peak_pending_bytes <= 64
    release.set()
    producer.join(5)
    recorder.close()
    assert done.is_set() and not failure
    assert recorder.pending_bytes == 0
    assert recorder.peak_pending_chunks == 2


def test_failed_writer_propagates_and_leaves_incomplete_step(tmp_path, monkeypatch):
    path = tmp_path / 'data.h5'
    original = Recorder._write_chunk
    def fail(self, handle, item):
        if item['kind'] == 'array':
            raise OSError('injected disk failure')
        return original(self, handle, item)
    monkeypatch.setattr(Recorder, '_write_chunk', fail)
    recorder = Recorder(RecordingPolicy(path, fields=('model',), chunk_bytes=16))
    try:
        recorder.record_accepted(accepted(), {'model': ArrayPayload(np.arange(40.), ('x',), 'm/s')})
    except RuntimeError as exc:
        assert 'writer failed' in str(exc)
    with pytest.raises(RuntimeError, match='writer failed') as failure:
        recorder.close()
    assert isinstance(failure.value.__cause__, OSError)
    assert not recorder._thread.is_alive()
    with h5py.File(path, 'r') as f:
        assert not f['steps/000000000001'].attrs['complete']


def test_detached_snapshot_does_not_retain_autograd_graph(tmp_path):
    import torch
    tensor = (torch.arange(12., dtype=torch.float64, requires_grad=True) * 2).reshape(3, 4)
    reference = weakref.ref(tensor)
    path = tmp_path / 'data.h5'
    with Recorder(RecordingPolicy(path, fields=('model',), chunk_bytes=16)) as recorder:
        recorder.record_accepted(accepted(), {'model': ArrayPayload(tensor, ('z', 'x'), 'm/s')})
        tensor.detach().fill_(99)
        del tensor
        gc.collect()
        assert reference() is None
    with h5py.File(path, 'r') as f:
        np.testing.assert_array_equal(f['steps/000000000001/model'][:], np.arange(12.).reshape(3, 4) * 2)


def test_existing_archive_is_never_silently_overwritten(tmp_path):
    path = tmp_path / 'data.h5'
    path.write_bytes(b'previous data')
    recorder = Recorder(RecordingPolicy(path))
    with pytest.raises(RuntimeError, match='writer failed'):
        recorder.close()
    assert path.read_bytes() == b'previous data'


def test_policy_and_payload_validation(tmp_path):
    with pytest.raises(ValueError):
        RecordingPolicy(tmp_path / 'x.h5', queue_capacity=0)
    with pytest.raises(ValueError):
        RecordingPolicy(tmp_path / 'x.h5', chunk_bytes=32, max_queue_bytes=16)
    with Recorder(RecordingPolicy(tmp_path / 'x.h5', fields=('model',))) as recorder:
        with pytest.raises(ValueError, match='axes'):
            recorder.record_accepted(accepted(), {'model': ArrayPayload(np.ones((2, 3)), ('x',), 'm/s')})


def test_reusable_pool_resident_byte_cap_with_variable_chunks(tmp_path):
    recorder = Recorder(RecordingPolicy(tmp_path / 'pool.h5', fields=('model',),
                                        chunk_bytes=64, max_queue_bytes=64))
    for step, size in enumerate((1, 9, 2, 17, 3), start=1):
        recorder.record_accepted(accepted(step), {'model': ArrayPayload(np.arange(float(size)), ('x',), 'm/s')})
    recorder.close()
    assert recorder.peak_buffer_bytes <= 64
    assert recorder.peak_pending_bytes <= 64


def test_noncontiguous_arrays_roundtrip(tmp_path):
    value = np.arange(5 * 7.).reshape(5, 7).T[:, ::-1]
    path = tmp_path / 'strided.h5'
    with Recorder(RecordingPolicy(path, fields=('model',), chunk_bytes=24)) as recorder:
        recorder.record_accepted(accepted(), {'model': ArrayPayload(value, ('z', 'x'), 'm/s')})
    with h5py.File(path, 'r') as f:
        np.testing.assert_array_equal(f['steps/000000000001/model'][:], value)


@pytest.mark.parametrize('view_kind', ['conjugate', 'negative'])
def test_lazy_tensor_views_are_rejected_without_materializing(tmp_path, monkeypatch, view_kind):
    import torch
    tensor = torch.ones((100, 100), dtype=torch.complex128)
    view = tensor.conj() if view_kind == 'conjugate' else tensor._neg_view()
    def forbidden(self):
        pytest.fail('recorder must reject lazy views before full tensor materialization')
    monkeypatch.setattr(torch.Tensor, 'resolve_conj', forbidden)
    monkeypatch.setattr(torch.Tensor, 'resolve_neg', forbidden)
    path = tmp_path / 'lazy.h5'
    with Recorder(RecordingPolicy(path, fields=('model',), chunk_bytes=64)) as recorder:
        with pytest.raises(ValueError, match='conjugate|negative'):
            recorder.record_accepted(accepted(), {'model': ArrayPayload(view, ('z', 'x'), 'Pa')})
        assert recorder.peak_buffer_bytes == 0
    with h5py.File(path, 'r') as f:
        assert len(f['steps']) == 0


def test_preparation_storage_does_not_scale_with_unselected_axis_length(tmp_path):
    import tracemalloc
    peaks = []
    with Recorder(RecordingPolicy(tmp_path / 'prep.h5', fields=('model',), chunk_bytes=64)) as recorder:
        for length in (100, 1000000):
            value = np.empty(length, dtype=np.float64)
            tracemalloc.start()
            array, indices, attrs = recorder._prepare('model', ArrayPayload(value, ('x',), 'm/s'))
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert np.shares_memory(array, value)
            assert isinstance(indices[0], range)
            peaks.append(peak)
    assert peaks[1] < peaks[0] + 32768


def test_oversized_coordinate_metadata_rejected_with_fixed_preparation_storage(tmp_path):
    import tracemalloc
    values = np.empty((1000000, 1), dtype=np.float64)
    with Recorder(RecordingPolicy(tmp_path / 'coordinates.h5', fields=('model',), chunk_bytes=64,
                                  sources=(1,))) as recorder:
        tracemalloc.start()
        with pytest.raises(ValueError, match='64 KiB metadata cap'):
            recorder._prepare('model', ArrayPayload(values, ('source', 'x'), 'm/s', sources=range(1000000)))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak < 4 * 1024 * 1024


@pytest.mark.cuda
@pytest.mark.skipif(not __import__('torch').cuda.is_available(), reason='requires actual CUDA runtime/device; not device-verified on CPU')
def test_cuda_pinned_event_roundtrip_selected_noncontiguous_and_no_whole_host_copy(tmp_path, monkeypatch):
    import torch
    base = torch.arange(3 * 5 * 7, dtype=torch.float64, device='cuda').reshape(3, 5, 7)
    value = torch.complex(base, base * 2).transpose(1, 2)
    expected = value[1:2].cpu().numpy().copy()
    original_cpu = torch.Tensor.cpu
    def no_cuda_cpu(self, *args, **kwargs):
        if self.device.type == 'cuda':
            pytest.fail('recorder attempted whole CUDA tensor .cpu()')
        return original_cpu(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'cpu', no_cuda_cpu)
    path = tmp_path / 'cuda.h5'
    recorder = Recorder(RecordingPolicy(path, fields=('pml_wavefield',), sources=(7,), chunk_bytes=64))
    recorder.record_accepted(accepted(), {'pml_wavefield': ArrayPayload(
        value, ('source_batch', 'z_padded', 'x_padded'), 'Pa', sources=(5, 7, 9))})
    # Completion at return preserves snapshot semantics even with immediate reuse.
    value.fill_(99)
    assert recorder.active_events == 0
    assert recorder.peak_active_events <= 2
    assert recorder.peak_buffer_bytes <= 128
    assert any(slot.owner is not None and slot.owner.is_pinned() for slot in recorder._slots)
    recorder.close()
    assert all(slot.owner is None and slot.event is None and not slot.references for slot in recorder._slots)
    with h5py.File(path, 'r') as f:
        ds = f['steps/000000000001/pml_wavefield']
        np.testing.assert_array_equal(ds[:], expected)
        assert ds.attrs['transfer_mode'] == 'cuda_pinned_event'


@pytest.mark.cuda
@pytest.mark.skipif(not __import__('torch').cuda.is_available(), reason='requires actual CUDA runtime/device; not device-verified on CPU')
def test_cuda_writer_waits_events_and_reuses_bounded_slots(tmp_path, monkeypatch):
    import torch
    original = Recorder._write_chunk
    writer_slots = []
    def inspected(self, handle, item):
        if item['kind'] == 'array' and item.get('slot') is not None:
            slot = item['slot']
            assert not slot.event_pending
            assert not slot.references
            writer_slots.append(id(slot))
        return original(self, handle, item)
    monkeypatch.setattr(Recorder, '_write_chunk', inspected)
    recorder = Recorder(RecordingPolicy(tmp_path / 'cuda-reuse.h5', fields=('model',), chunk_bytes=64))
    tensor = torch.arange(100., dtype=torch.float64, device='cuda')
    for step in range(1, 4):
        recorder.record_accepted(accepted(step), {'model': ArrayPayload(tensor, ('x',), 'm/s')})
    recorder.close()
    assert len(writer_slots) > 2
    assert len(set(writer_slots)) <= 2
    assert recorder.peak_pending_chunks <= 2
    assert recorder.peak_pending_bytes <= 128
    assert recorder.peak_buffer_bytes <= 128
    assert recorder.peak_active_events <= 2
    assert recorder.active_events == 0


@pytest.mark.cuda
@pytest.mark.skipif(not __import__('torch').cuda.is_available(), reason='requires actual CUDA runtime/device; not device-verified on CPU')
def test_cuda_disk_failure_drains_events_and_propagates_without_stuck_producer(tmp_path, monkeypatch):
    import torch
    original = Recorder._write_chunk
    def fail(self, handle, item):
        if item['kind'] == 'array':
            raise OSError('injected CUDA archival disk failure')
        return original(self, handle, item)
    monkeypatch.setattr(Recorder, '_write_chunk', fail)
    recorder = Recorder(RecordingPolicy(tmp_path / 'cuda-failed.h5', fields=('model',), chunk_bytes=64))
    source = torch.arange(100., dtype=torch.float64, device='cuda')
    try:
        recorder.record_accepted(accepted(), {'model': ArrayPayload(source, ('x',), 'm/s')})
    except RuntimeError as exc:
        assert 'writer failed' in str(exc)
    with pytest.raises(RuntimeError, match='writer failed'):
        recorder.close()
    assert not recorder._thread.is_alive()
    assert recorder.active_events == 0
    assert all(not slot.references and slot.owner is None for slot in recorder._slots)


def test_cpu_then_cpu_slots_are_released_and_have_no_events(tmp_path):
    recorder = Recorder(RecordingPolicy(tmp_path / 'cpu-clean.h5', fields=('model',), chunk_bytes=32))
    recorder.record_accepted(accepted(), {'model': ArrayPayload(np.arange(10.), ('x',), 'm/s')})
    recorder.close()
    assert recorder.active_events == 0
    assert recorder.peak_active_events == 0
    assert all(slot.raw is None and slot.owner is None and slot.event is None for slot in recorder._slots)


def test_host_control_flow_completion_failure_wakes_producer_and_preserves_ownership(tmp_path, monkeypatch):
    """Synthetic event tests host error handling only; this is not CUDA evidence."""
    original = Recorder._get_slot
    token = object()
    class FailedCompletion:
        def synchronize(self):
            raise RuntimeError('injected completion failure (host control-flow probe)')
    def failed_slot(self, count, dtype, cuda_device=None):
        slot = original(self, count, dtype, cuda_device)
        slot.event = FailedCompletion()
        slot.event_pending = True
        slot.references = (token,)
        return slot
    monkeypatch.setattr(Recorder, '_get_slot', failed_slot)
    recorder = Recorder(RecordingPolicy(tmp_path / 'event-failed.h5', fields=('model',), chunk_bytes=16))
    finished = threading.Event()
    caught = []
    def produce():
        try:
            recorder.record_accepted(accepted(), {'model': ArrayPayload(np.arange(40.), ('x',), 'm/s')})
        except RuntimeError as exc:
            caught.append(exc)
        finally:
            finished.set()
    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    assert finished.wait(5), 'producer remained blocked after completion failure'
    producer.join(1)
    with pytest.raises(RuntimeError, match='writer failed'):
        recorder.close()
    assert caught and not recorder._thread.is_alive()
    assert any(slot.references == (token,) for slot in recorder._slots)
    assert any(slot.raw is not None for slot in recorder._slots)
    with h5py.File(tmp_path / 'event-failed.h5', 'r') as f:
        assert not f['steps/000000000001'].attrs['complete']


def test_host_control_flow_slot_does_not_clear_before_event_completion():
    """Synthetic event checks ownership state transitions, not device correctness."""
    from simplefwi.fwi2d.recording import _HostSlot
    calls = []
    class Completion:
        def synchronize(self):
            calls.append('wait')
    slot = _HostSlot(raw=np.empty(4, dtype=np.uint8), event=Completion(),
                     event_pending=True, references=(object(),))
    with pytest.raises(RuntimeError, match='before completion'):
        slot.clear()
    slot.wait()
    assert calls == ['wait'] and not slot.references and not slot.event_pending
    slot.clear()
    assert slot.raw is None and slot.event is None


def test_small_accepted_diagnostics_persist_when_arrays_skipped(tmp_path):
    path = tmp_path / 'diagnostics.h5'
    diagnostics = {'optimizer_class': 'LBFGS', 'learning_rates': [1.], 'gradient_norm': .5}
    with Recorder(RecordingPolicy(path, cadence=5)) as recorder:
        recorder.record_accepted(accepted(1, closure_evaluations=3, diagnostics=diagnostics), {})
        diagnostics['learning_rates'][0] = 99.
    with h5py.File(path, 'r') as f:
        attrs = f['steps/000000000001'].attrs
        assert attrs['closure_evaluations'] == 3
        saved = json.loads(attrs['diagnostics'])
        assert saved['learning_rates'] == [1.]
        assert saved['gradient_norm'] == .5


def test_oversized_or_nonfinite_diagnostics_rejected_before_recording(tmp_path):
    path = tmp_path / 'invalid-diagnostics.h5'
    with Recorder(RecordingPolicy(path)) as recorder:
        for diagnostics in ({'large': 'x' * 65537}, {'value': float('nan')}):
            with pytest.raises(ValueError):
                recorder.record_accepted(accepted(diagnostics=diagnostics), {})
        with pytest.raises(ValueError, match='closure_evaluations'):
            recorder.record_accepted(accepted(closure_evaluations=-1), {})
    with h5py.File(path, 'r') as f:
        assert len(f['steps']) == 0


def test_later_slot_allocation_failure_preserves_original_error_and_accounting(tmp_path, monkeypatch):
    """Host allocation cleanup regression; does not require or verify CUDA."""
    recorder = Recorder(RecordingPolicy(tmp_path / 'allocation-failed.h5', chunk_bytes=32))
    dtype = np.dtype('float64')
    first = recorder._get_slot(4, dtype)
    failure = MemoryError('injected second-slot allocation failure')
    original_empty = np.empty
    producer_thread = threading.current_thread()
    def fail_allocation(*args, **kwargs):
        if threading.current_thread() is producer_thread:
            raise failure
        return original_empty(*args, **kwargs)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(np, 'empty', fail_allocation)
            with pytest.raises(MemoryError) as caught:
                recorder._get_slot(4, dtype)
        assert caught.value is failure
        assert recorder._buffer_count == 1
        assert len(recorder._slots) == 1 and recorder._slots[0] is first
        assert recorder._pool == []
        assert first.raw is not None
        assert recorder.pending_bytes == 0 and recorder.active_events == 0
        second = recorder._get_slot(4, dtype)
        assert second is not first and second.raw is not None
        assert recorder._buffer_count == 2 and len(recorder._slots) == 2
        assert recorder.peak_buffer_bytes <= 64
    finally:
        recorder.close()
