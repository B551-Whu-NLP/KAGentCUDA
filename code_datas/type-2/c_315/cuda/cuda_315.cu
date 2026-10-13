#include <stdio.h>
#include <cuda_runtime.h>

#define N 512
#define Size 1024

// CUDA kernel
__global__ void binarize_input_kernel(float *input, int n, int size, float *binary) {
    int s = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (s >= size)
        return;

    int i = 0;
    float mean = 0;

    for (i = 0; i < n; ++i) {
        mean += fabs(input[i * size + s]);
    }

    mean = mean / n;

    for (i = 0; i < n; ++i) {
        binary[i * size + s] = (input[i * size + s] > 0) ? mean : -mean;
    }
}

int main() {
    // Allocate device memory
    float *d_input, *d_binary;

    cudaMalloc((void **)&d_input, N * Size * sizeof(float));
    cudaMalloc((void **)&d_binary, N * Size * sizeof(float));

    // Set grid and block sizes
    dim3 blockSize(256);  // You may adjust the block size
    dim3 gridSize((N * Size + blockSize.x - 1) / blockSize.x);

    // Launch the kernel
    binarize_input_kernel<<<gridSize, blockSize>>>(d_input, N, Size, d_binary);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Check for errors during the kernel launch
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_binary);

    return 0;
}
 
