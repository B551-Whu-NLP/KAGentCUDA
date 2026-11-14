#include <stdio.h>

__global__ void MulMatrixOnGPU(float* A, float* B, float* C, int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < nx && j < ny) {
        float sum = 0.0;

        for (int k = 0; k < nx; k++) {
            sum += A[i * nx + k] * B[k * nx + j];
        }

        C[i * nx + j] = sum;
    }
}

int main() {
    // Set your desired parameters
    int nx = 256;  // Set your desired value for nx
    int ny = 128;  // Set your desired value for ny

    // Allocate memory on the host
    float* h_A = nullptr;  // Add initialization or copy data to h_A
    float* h_B = nullptr;  // Add initialization or copy data to h_B
    float* h_C = new float[nx * ny];  // Allocate memory for the result

    // Allocate memory on the device
    float* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeof(float) * nx * nx);  // Add appropriate size
    cudaMalloc((void**)&d_B, sizeof(float) * nx * nx);  // Add appropriate size
    cudaMalloc((void**)&d_C, sizeof(float) * nx * nx);  // Add appropriate size

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((nx + 15) / 16, (ny + 15) / 16, 1);
    dim3 blockSize(16, 16, 1);

    // Launch the CUDA kernel for matrix multiplication
    MulMatrixOnGPU<<<gridSize, blockSize>>>(d_A, d_B, d_C, nx, ny);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_C;

    return 0;
}
