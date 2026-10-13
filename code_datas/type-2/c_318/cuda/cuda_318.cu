#include <stdio.h>
#include <cuda_runtime.h>

__global__ void mean_kernel(float *x, int batch, int filters, int spatial, float *mean) {
    float scale = 1.f / (batch * spatial);
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters)
        return;

    int j, k;
    mean[i] = 0;

    for (j = 0; j < batch; ++j) {
        for (k = 0; k < spatial; ++k) {
            int index = j * filters * spatial + i * spatial + k;
            mean[i] += x[index];
        }
    }

    mean[i] *= scale;
}

int main() {
    // Set the dimensions
    int batch = 64;      // Set the appropriate values
    int filters = 128;   // Set the appropriate values
    int spatial = 256;   // Set the appropriate values

    // Allocate device memory
    float *d_x, *d_mean;
    cudaMalloc((void **)&d_x, batch * filters * spatial * sizeof(float));
    cudaMalloc((void **)&d_mean, filters * sizeof(float));

    // Set grid and block sizes
    dim3 blockSize(256);  // You may adjust the block size
    dim3 gridSize((filters + blockSize.x - 1) / blockSize.x, 1);

    // Launch the kernel
    mean_kernel<<<gridSize, blockSize>>>(d_x, batch, filters, spatial, d_mean);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Check for errors during the kernel launch
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_mean);

    return 0;
}
 
