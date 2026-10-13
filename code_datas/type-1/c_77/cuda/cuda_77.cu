#include <stdio.h>

// CUDA kernel for matrix addition
__global__ void addMatrixGPU(float* a, float* b, float* c, int N) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i < N) && (j < N)) {
        int idx = i * N + j;
        a[idx] = b[idx] + c[idx];
    }
}

int main() {
    // Set your desired matrix dimensions
    int N = 512;

    // Allocate memory on the host
    float* h_a = (float*)malloc(N * N * sizeof(float));
    float* h_b = (float*)malloc(N * N * sizeof(float));
    float* h_c = (float*)malloc(N * N * sizeof(float));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    float* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, N * N * sizeof(float));
    cudaMalloc((void**)&d_b, N * N * sizeof(float));
    cudaMalloc((void**)&d_c, N * N * sizeof(float));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((N + 15) / 16, (N + 15) / 16);
    dim3 blockSize(16, 16);

    // Launch the CUDA kernel for matrix addition
    addMatrixGPU<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
