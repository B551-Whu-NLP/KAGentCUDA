#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel
__global__ void backward_avgpool_layer_kernel(int n, int w, int h, int c, float *in_delta, float *out_delta) {
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n)
        return;

    int k = id % c;
    id /= c;
    int b = id;
    int i;
    int out_index = (k + c * b);

    for (i = 0; i < w * h; ++i) {
        int in_index = i + h * w * (k + b * c);
        in_delta[in_index] += out_delta[out_index] / (w * h);
    }
}

int main() {
    // Define the dimensions
    int n = 1024;  // You need to set the appropriate value for n
    int w = 28;    // Set the appropriate values for w, h, and c
    int h = 28;
    int c = 3;

    // Allocate device memory
    float *d_in_delta, *d_out_delta;

    cudaMalloc((void **)&d_in_delta, n * w * h * c * sizeof(float));
    cudaMalloc((void **)&d_out_delta, n * c * sizeof(float));

    // Set grid and block sizes
    dim3 blockSize(256);  // You may adjust the block size
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, 1);

    // Launch the kernel
    backward_avgpool_layer_kernel<<<gridSize, blockSize>>>(n, w, h, c, d_in_delta, d_out_delta);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Check for errors during the kernel launch
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    // Cleanup
    cudaFree(d_in_delta);
    cudaFree(d_out_delta);

    return 0;
}
 
