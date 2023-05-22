import os

import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from PIL import Image


class InvertImage:
    with open(os.path.join("app", "invert.cu"), "r") as kernel:
        mod = SourceModule(kernel.read())
    invert_kernel = mod.get_function("invertKernel")

    def __call__(self, input_path: str, output_path: str):
        input_image = Image.open(input_path)
        input_array = np.array(input_image).astype(np.uint8)
        output_array = np.empty_like(input_array)
        width, height, channels = input_array.shape
        block = (256, 1, 1)
        grid = ((width * height * channels + block[0] - 1) // block[0], 1)
        self.invert_kernel(
            drv.In(input_array),
            drv.Out(output_array),
            np.int32(width),
            np.int32(height),
            np.int32(channels),
            block=block,
            grid=grid,
        )
        output_image = Image.fromarray(output_array)
        output_image.save(output_path)
