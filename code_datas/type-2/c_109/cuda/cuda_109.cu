#include <stdio.h>

__global__ void mxm_1d(double* a, const int m, double* b, const int n, double* c, const int p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m) {
        for (int k = 0; k < p; k++) {
            double s = 0.0;

            for (int j = 0; j < n; j++) {
                s += a[j * m + i] * b[k * n + j];
            }

            c[k * m + i] = s;
        }
    }
}

int main() {
    // Set your desired parameters
    int m = 256;  // Set your desired value for m
    int n = 128;  // Set your desired value for n
    int p = 64;   // Set your desired value for p

    // Allocate memory on the host
    double* h_a = nullptr;  // Add initialization or copy data to h_a
    double* h_b = nullptr;  // Add initialization or copy data to h_b

    // Allocate memory on the device
    double* d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, sizeof(double));  // Add appropriate size
    cudaMalloc((void**)&d_b, sizeof(double));  // Add appropriate size
    cudaMalloc((void**)&d_c, sizeof(double));  // Add appropriate size

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((m + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);

    // Launch the CUDA kernel for matrix-matrix multiplication
    mxm_1d<<<gridSize, blockSize>>>(d_a, m, d_b, n, d_c, p);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    // Add code to free host memory if needed

    return 0;
}
