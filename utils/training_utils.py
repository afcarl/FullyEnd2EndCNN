import h5py
import numpy as np
try:
    from Queue import Queue
    from Queue import Empty
except ImportError:
    from queue import Queue
    from queue import Empty
from ..utils import logging_utils as lu


def _mmap_h5(path, h5path):
    """Memmap HDF5 for faster indexing"""

    with h5py.File(path) as f:
        ds = f[h5path]
        # We get the dataset address in the HDF5 field.
        offset = ds.id.get_offset()
        # We ensure we have a non-compressed contiguous array.
        assert ds.chunks is None
        assert ds.compression is None
        assert offset > 0
        dtype = ds.dtype
        shape = ds.shape
    arr = np.memmap(path, mode="r", shape=shape, offset=offset, dtype=dtype)
    return arr


def producer(queue, settings, iteration, stop_event):
    """Producer for threading to create batches of features and targets"""

    X = _mmap_h5(settings.hogs_file, "training/it_%s/samples_feat" % iteration)
    y = _mmap_h5(settings.hogs_file, "training/it_%s/normed_delta_params" % iteration)
    npts = X.shape[0]
    while not stop_event.is_set():
        start = np.random.randint(0, npts - settings.batch_size - 1)
        end = start + settings.batch_size
        queue.put((np.array(X[start: end]), np.array(y[start: end])))


def close_thread(thread, queue, stop_event):
    """Utility to close thread and avoid bloating/slowdown"""

    stop_event.set()
    try:
        while True:
            queue.get_nowait()
    except Empty:
        del queue
    thread.join()
    lu.print_green("Close thread")
