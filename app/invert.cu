extern "C" {
    __global__ void invertKernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < width * height * channels) {
            output[idx] = 255 - input[idx];
        }
    }
}
