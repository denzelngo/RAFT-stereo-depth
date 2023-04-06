import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np


SHAPE_IN = (320, 200)

# Returns a numpy buffer of shape (num_images, 1, 28, 28)
def load_image(img_path):
    img_l = Image.open(img_path).convert('RGB').resize(SHAPE_IN)
    img_r = Image.open(img_path.replace('left', 'right')).convert('RGB').resize(SHAPE_IN)

    img_l = np.array(img_l)[:,:,::-1].astype(np.float32)
    img_r = np.array(img_r)[:,:,::-1].astype(np.float32)
    
    img_l = np.transpose(img_l, [2,0,1])
    img_r = np.transpose(img_r, [2,0,1])

    img_l = np.ascontiguousarray(img_l)
    img_r = np.ascontiguousarray(img_r)

    return img_l, img_r


class RAFTEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data, cache_file, batch_size=4):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.img_files = [os.path.join(training_data, 'left', i) for i in os.listdir(training_data + '/left')]
        self.batch_size = batch_size
        self.current_index = 0
        self.shape = (batch_size, 3, SHAPE_IN[1], SHAPE_IN[0])

        # Allocate enough memory for a whole batch.
        self.device_input_l = cuda.mem_alloc(trt.volume(self.shape)*trt.float32.itemsize)
        self.device_input_r = cuda.mem_alloc(trt.volume(self.shape)*trt.float32.itemsize)

        def load_batches():
            start = 0
            for i in range(len(self.img_files)):
                yield self.read_batch_file(self.img_files[start:start+self.batch_size])
                start += self.batch_size
        self.batches = load_batches()

    def read_batch_file(self, file_paths):
        batch_l = []
        batch_r = []
        for img_path in file_paths:
            img_l, img_r = load_image(img_path)
            batch_l.append(img_l)
            batch_r.append(img_r)
        batch_l = np.array(batch_l)
        batch_r = np.array(batch_r)
        return batch_l, batch_r


    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.img_files):
            return None

        batch = next(self.batches)
        cuda.memcpy_htod(self.device_input_l, batch[0].ravel())
        cuda.memcpy_htod(self.device_input_r, batch[1].ravel())
        self.current_index += self.batch_size
        return [self.device_input_l, self.device_input_r]


    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
