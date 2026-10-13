#include <stdio.h>
#include <math.h>

__global__ void fabsf_clamp_kernel(int N, float* X, int INCX, float clamp_min, float clamp_max) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        if (X[i * INCX] >= 0)
            X[i * INCX] = fminf(clamp_max, fmaxf(clamp_min, X[i * INCX]));
        else
            X[i * INCX] = fminf(-clamp_min, fmaxf(-clamp_max, X[i * INCX]));
    }
}

int main() {
    // Set your desired parameters
    int N = 1024;  // Set your desired value for N
    float clamp_min = -1.0f;  // Set your desired value for clamp_min
    float clamp_max = 1.0f;   // Set your desired value for clamp_max

    // Allocate memory on the host
    float* h_X = nullptr;  // Add initialization or copy data to h_X

    // Allocate memory on the device
    float* d_X;
    cudaMalloc((void**)&d_X, sizeof(float));  // Add appropriate size

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((N + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);

    // Launch the CUDA kernel for fabsf_clamp
    fabsf_clamp_kernel<<<gridSize, blockSize>>>(N, d_X, 1, clamp_min, clamp_max);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_X);

    // Free host memory
    // Add code to free host memory if needed

    return 0;
}
