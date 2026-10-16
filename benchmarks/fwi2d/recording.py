"""Bounded-source recording off/on microbenchmark; see guides/outputs.md."""
import sys, time, tempfile, resource, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from simplefwi.fwi2d.recording import ArrayPayload, Recorder, RecordingPolicy
mode = sys.argv[1]
temporary = tempfile.TemporaryDirectory()
path = Path(temporary.name) / 'snapshots.h5'
array = np.ones((4, 96, 96), dtype=np.complex128) * (1 + 2j)
start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
recorder = Recorder(RecordingPolicy(path, fields=('pml_wavefield',),
                     chunk_bytes=256 * 1024)) if mode == 'on' else None
start = time.perf_counter()
checksum = 0.
for step in range(1, 17):
    array *= complex(1.00001, .00001)
    checksum += float(np.vdot(array, array).real)
    if recorder:
        recorder.record_accepted(SimpleNamespace(accepted=True, accepted_step=step, loss=checksum), {
            'pml_wavefield': ArrayPayload(array, ('source_batch', 'z_padded', 'x_padded'),
                'Pa', origin=(0., 0.), sources=(0, 1, 2, 3))})
if recorder:
    recorder.close()
print(json.dumps({'mode': mode, 'runtime_s': time.perf_counter() - start,
    'rss_baseline_KiB': start_rss,
    'peak_host_RSS_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    'peak_queue_bytes': recorder.peak_pending_bytes if recorder else 0,
    'peak_pool_bytes': recorder.peak_buffer_bytes if recorder else 0,
    'peak_queue_chunks': recorder.peak_pending_chunks if recorder else 0,
    'file_bytes': path.stat().st_size if recorder else 0, 'checksum': checksum,
    'device_peak_bytes': None, 'payload_bytes': array.nbytes * 16}))
