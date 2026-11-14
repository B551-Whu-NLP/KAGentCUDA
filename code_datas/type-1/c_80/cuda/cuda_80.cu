#include <stdio.h>

// CUDA kernel for matrix addition
__global__ void AddMatrixOnGPU(float* A, float* B, float* C, int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = i * nx + j;

    if (i < nx && j < ny) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Set your desired matrix dimensions
    int nx = 512;
    int ny = 512;

    // Allocate memory on the host
    float* h_A = (float*)malloc(nx * ny * sizeof(float));
    float* h_B = (float*)malloc(nx * ny * sizeof(float));
    float* h_C = (float*)malloc(nx * ny * sizeof(float));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, nx * ny * sizeof(float));
    cudaMalloc((void**)&d_B, nx * ny * sizeof(float));
    cudaMalloc((void**)&d_C, nx * ny * sizeof(float));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((nx + 15) / 16, (ny + 15) / 16);
    dim3 blockSize(16, 16);

    // Launch the CUDA kernel for matrix addition
    AddMatrixOnGPU<<<gridSize, blockSize>>>(d_A, d_B, d_C, nx, ny);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
