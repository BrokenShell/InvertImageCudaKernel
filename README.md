# InvertImage CUDA Kernel

This repository contains a CUDA kernel for inverting the colors of an image. The CUDA kernel leverages the power of GPU parallelism to efficiently perform the inversion operation on the image.

## Requirements

To use this CUDA kernel, the following dependencies are required:

- Python (version 3 or higher, 3.11+ recommended)
- NumPy (Python package)
- PyCUDA (Python package)
- PIL (Python Imaging Library)
- CUDA Enabled GPU

## Installation

1. Clone or download this repository to your local machine.

2. Make sure you have the required dependencies installed. You can install them using the following command:

```shell
pip install -r requirements.txt
```

## Usage

Once you have installed the package, you can use the `InvertImage` class to invert images using the CUDA kernel. The `InvertImage` class provides a callable object that takes an input image path and an output image path as arguments. The input image is inverted using the CUDA kernel, and the result is saved to the output image path.

Here's an example of how to use the `InvertImage` class:

```python
# main.py
from app.invert import InvertImage

invert_image = InvertImage()
invert_image("input.jpeg", "output.jpeg")
```

### Input Image
![Input Image](input.jpeg)

### Output Image
![Output Image](output.jpeg)

## Explanation

The CUDA kernel code can be found in the `invert.cu` file. It defines a kernel function `invertKernel` that performs the inversion operation on the input image.

The `InvertImage` class initializes the CUDA kernel by loading the kernel code from the `invert.cu` file using the `pycuda.compiler.SourceModule` class. It also sets up the necessary configurations for executing the kernel.

When the `__call__` method of the `InvertImage` class is invoked, it performs the following steps:

1. Opens the input image using the PIL library and converts it to a NumPy array.

2. Allocates memory for the output array, which will store the inverted image.

3. Determines the dimensions of the input array (width, height, and channels).

4. Sets up the block and grid configurations for executing the CUDA kernel.

5. Invokes the CUDA kernel, passing the input and output arrays, along with the dimensions, as arguments.

6. Converts the output array back to an image using the PIL library and saves it to the specified output path.
