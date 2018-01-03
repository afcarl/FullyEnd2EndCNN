import numpy as np
import h5py
import cv2
from numba import jit

try:
    from Queue import Queue
    from Queue import Empty
except ImportError:
    from queue import Queue
    from queue import Empty
import threading
from functools import partial

import torch
from torch.autograd import Variable


def identity(img, pts0, init0):

    return img, pts0, init0


@jit
def hflip(img, pts0, init0):

    pts0[:, 0] = img.shape[-1] - pts0[:, 0]

    return img[:, ::-1], pts0, init0


def random_rotation(img, max_angle=10):

    h, w = img.shape[:2]
    rot_center = (w / 2, h / 2)

    angle = np.random.uniform(-max_angle, max_angle)
    rotM = cv2.getRotationMatrix2D(rot_center, angle, 1)
    img = cv2.warpAffine(img, rotM, (w, h))
    img = img.reshape((h, w))

    return img


def random_translation(img, max_x=10, max_y=10):

    h, w = img.shape[:2]

    tr_x = np.random.uniform(-max_x, max_x)
    tr_y = np.random.uniform(-max_y, max_y)

    trM = np.float32([[1, 0, tr_x],[0, 1, tr_y]])
    img = cv2.warpAffine(img, trM, (w, h))

    img = img.reshape((h, w))

    return img


def random_zoom(img, max_x, max_y):

    h, w = img.shape[:2]

    tr_x = np.random.uniform(-max_x, max_x)
    tr_y = np.random.uniform(-max_y, max_y)

    trM = np.float32([[1, 0, tr_x],[0, 1, tr_y]])
    img = cv2.warpAffine(img, trM, (w, h))

    img = img.reshape((h, w))

    return img


def random_crop(img, min_crop_size, max_crop_size):

    h, w = img.shape[:2]

    crop_size_x = np.random.randint(min_crop_size, max_crop_size)
    crop_size_y = np.random.randint(min_crop_size, max_crop_size)
    pos_x = np.random.randint(0, 1 + w - crop_size_x)  # +1 because npy slices
    pos_y = np.random.randint(0, 1 + h - crop_size_y)  # are non inclusive
    img = img[pos_y:pos_y + crop_size_y,
              pos_x:pos_x + crop_size_x, :]
    # Resize
    img = cv2.resize(img, (w, h))

    return img


def random_blur(img, pts0, init0, max_kernel_size=2):

    kernel_size = np.random.randint(1, max_kernel_size)
    blur = (kernel_size, kernel_size)
    img = cv2.blur(img, blur)

    return img, pts0, init0


def random_dilate(img, pts0, init0, max_kernel_size=2):

    kernel_size = np.random.randint(1, max_kernel_size)
    kernel = np.ones((kernel_size,
                      kernel_size), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    return img, pts0, init0


def random_erode(img, pts0, init0, max_kernel_size=2):

    kernel_size = np.random.randint(1, max_kernel_size)
    kernel = np.ones((kernel_size,
                      kernel_size), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)

    return img, pts0, init0


def histogram_equalization(img, pts0, init0):

    return cv2.equalizeHist(img), pts0, init0


@jit
def random_occlusion(img, pts0, init0, size_x=1, size_y=1):

    h, w = img.shape

    # Random pos
    x_start = np.random.randint(0, w - size_x)
    y_start = np.random.randint(0, h - size_y)
    img[y_start: y_start + size_y, x_start: x_start + size_x] = 0

    return img, pts0, init0


def pixel_dropout(img, pts0, init0, mask=None):

    assert mask is not None

    return img * mask, pts0, init0


@jit
def invert(img, pts0, init0):

    return 255 - img, pts0, init0


def equalize_invert(img, pts0, init0):

    return 255 - cv2.equalizeHist(img), pts0, init0


def _mmap_h5(path, h5path):

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


def batch_generator(queue, settings, stop_event, d_augmentation):

    img0 = _mmap_h5("data/processed/CNN_database.h5", "training/img0")
    pts0 = _mmap_h5("data/processed/CNN_database.h5", "training/pts0")
    init0 = _mmap_h5("data/processed/CNN_database.h5", "training/init0")

    print "Starting filling queues"

    # Get the number of samples
    npts = img0.shape[0]

    num_batches = npts // settings.batch_size
    list_batches = np.array_split(np.arange(npts), num_batches)

    idx_batch = 0

    while not stop_event.is_set():

        # Reset idx_batch if needed
        if idx_batch > len(list_batches) - 1:
            idx_batch = 0

        start, end = list_batches[idx_batch][0], list_batches[idx_batch][-1] + 1

        # start = np.random.randint(0, npts - settings.batch_size - 1)
        # end = start + settings.batch_size

        img0_arr = np.array(img0[start: end])
        pts0_arr = np.array(pts0[start: end])
        init0_arr = np.array(init0[start: end])

        num_elem = settings.batch_size
        num_chunks = len(d_augmentation.keys())
        list_chunks = np.array_split(np.arange(num_elem), num_chunks)

        for i in np.random.permutation(len(list_chunks)):

            # Get the function whose name matches the string name
            aug_function_as_str = d_augmentation.keys()[i]
            aug_function = globals()[aug_function_as_str]

            # Some function may have key word arguments, provided in d_augmentation
            # Create a partial function to automatically use those key word arguments
            kwargs = d_augmentation[aug_function_as_str]

            # Special case : pixel dropout (generate the same mask for all images for speed reasons)
            if aug_function_as_str == "pixel_dropout":
                mask = np.random.binomial(1, 1 - kwargs["dropout_fraction"], (img0_arr.shape[1:])).astype(np.uint8)
                kwargs = {"mask": mask}

            if kwargs != {}:
                aug_function = partial(aug_function, **kwargs)

            # Select images (i.e. idxs) on which we want to apply the augmentation
            idxs = list_chunks[i]

            for idx in idxs:
                img0_arr[idx], pts0_arr[idx], init0_arr[idx] = aug_function(img0_arr[idx],
                                                                            pts0_arr[idx],
                                                                            init0_arr[idx])

        # Select a random init
        idx_init = np.random.randint(0, init0_arr.shape[1])
        init0_arr = init0_arr[:, idx_init, :, :]
        # Get the normalized delta0
        delta0_norm_arr = (pts0_arr - init0_arr).reshape(-1, 42)
        delta0_norm_arr -= settings.delta0_mean
        delta0_norm_arr /= settings.delta0_std

        queue.put((img0_arr.astype(np.float32) / 255., pts0_arr, init0_arr, delta0_norm_arr))

        # Increment idx_batch
        idx_batch += 1


class AugDataGenerator(object):
    """
    """

    def __init__(self,
                 settings,
                 d_augmentation={}):

        self.settings = settings
        self.d_augmentation = d_augmentation

    def create_queue(self):

        self.queue = Queue(maxsize=self.settings.queue_max_size)
        self.stop_event = threading.Event()
        self.load_thread = threading.Thread(target=batch_generator, args=(self.queue,
                                                                          self.settings,
                                                                          self.stop_event,
                                                                          self.d_augmentation))

        self.load_thread.start()

        return self.queue

    def stop_queue(self):

        self.stop_event.set()
        try:
            while True:
                self.queue.get_nowait()
        except Empty:
            del self.queue
        self.load_thread.join()


if __name__ == '__main__':

    import ipdb
    ipdb.set_trace()
