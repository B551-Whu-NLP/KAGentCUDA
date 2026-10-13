#include <stdio.h>

// CUDA kernel for naive matrix multiplication
__global__ void naive_sgemm_kernel(float* C, float* A, float* B, long size) {
    const long i = blockIdx.x * blockDim.x + threadIdx.x;
    const long j = blockIdx.y * blockDim.y + threadIdx.y;

    float val = 0.0;

    if (i >= size || j >= size)
        return;

    for (long k = 0; k < size; k++) {
        val += A[i * size + k] * B[k * size + j];
    }

    C[i * size + j] += val;
}

int main() {
    // Set your desired parameters
    long size = 512;

    // Allocate memory on the host
    float* h_C = (float*)malloc(size * size * sizeof(float));
    float* h_A = (float*)malloc(size * size * sizeof(float));
    float* h_B = (float*)malloc(size * size * sizeof(float));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    float* d_C, * d_A, * d_B;
    cudaMalloc((void**)&d_C, size * size * sizeof(float));
    cudaMalloc((void**)&d_A, size * size * sizeof(float));
    cudaMalloc((void**)&d_B, size * size * sizeof(float));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((size + 15) / 16, (size + 15) / 16, 1);
    dim3 blockSize(16, 16, 1);

    // Launch the CUDA kernel for naive matrix multiplication
    naive_sgemm_kernel<<<gridSize, blockSize>>>(d_C, d_A, d_B, size);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_C);
    cudaFree(d_A);
    cudaFree(d_B);

    // Free host memory
    free(h_C);
    free(h_A);
    free(h_B);

    return 0;
}
