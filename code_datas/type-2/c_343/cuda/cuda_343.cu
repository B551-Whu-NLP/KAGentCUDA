#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

__global__ void reorg_kernel(int N, float *x, int w, int h, int c, int batch, int stride, int forward, float *out) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i >= N)
        return;

    int in_index = i;
    int in_w = i % w;
    i = i / w;
    int in_h = i % h;
    i = i / h;
    int in_c = i % c;
    i = i / c;
    int b = i % batch;

    int out_c = c / (stride * stride);
    int c2 = in_c % out_c;
    int offset = in_c / out_c;
    int w2 = in_w * stride + offset % stride;
    int h2 = in_h * stride + offset / stride;

    int out_index = w2 + w * stride * (h2 + h * stride * (c2 + out_c * b));

    if (forward)
        out[out_index] = x[in_index];
    else
        out[in_index] = x[out_index];
}

int main() {
    const int N = 1024; // Adjust the size based on your data
    const int w = 64;   // Adjust the width based on your data
    const int h = 64;   // Adjust the height based on your data
    const int c = 3;    // Adjust the channels based on your data
    const int batch = 4; // Adjust the batch size based on your data
    const int stride = 2; // Adjust the stride based on your requirement

    float *x, *out;

    size_t size = N * sizeof(float);

    // Allocate host memory
    x = (float *)malloc(size);
    out = (float *)malloc(size);

    // Initialize host data (you may need to modify this based on your use case)
    for (int i = 0; i < N; ++i) {
        x[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_x, *d_out;
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_out, size);

    // Copy data from host to device
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((N + block_size.x - 1) / block_size.x);

    // Launch the kernel
    reorg_kernel<<<grid_size, block_size>>>(N, d_x, w, h, c, batch, stride, 1, d_out); // Set the last parameter to 1 for forward, 0 for backward

    // Copy data from device to host
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    // Free device and host memory
    cudaFree(d_x);
    cudaFree(d_out);
    free(x);
    free(out);

    return 0;
}
 
