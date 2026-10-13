#include <stdio.h>

__global__ void mmul(const float* A, const float* B, float* C, int r1, int c1, int r2, int c2) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if ((idx < c2) && (idy < c1)) {
        float temp = 0;

        for (int i = 0; i < c1; i++)
            temp += A[idy * c1 + i] * B[i * c2 + idx];

        C[idy * c2 + idx] = temp;
    }
}

int main() {
    // Set your desired parameters
    int r1 = 256;  // Set your desired value for r1
    int c1 = 128;  // Set your desired value for c1
    int r2 = 128;  // Set your desired value for r2
    int c2 = 64;   // Set your desired value for c2

    // Allocate memory on the host
    float* h_A = nullptr;  // Add initialization or copy data to h_A
    float* h_B = nullptr;  // Add initialization or copy data to h_B
    float* h_C = new float[r1 * c2];  // Allocate memory for the result

    // Allocate memory on the device
    float* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeof(float) * r1 * c1);  // Add appropriate size
    cudaMalloc((void**)&d_B, sizeof(float) * c1 * c2);  // Add appropriate size
    cudaMalloc((void**)&d_C, sizeof(float) * r1 * c2);  // Add appropriate size

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((c2 + 15) / 16, (r1 + 15) / 16, 1);
    dim3 blockSize(16, 16, 1);

    // Launch the CUDA kernel for matrix multiplication
    mmul<<<gridSize, blockSize>>>(d_A, d_B, d_C, r1, c1, r2, c2);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_C;

    return 0;
}
