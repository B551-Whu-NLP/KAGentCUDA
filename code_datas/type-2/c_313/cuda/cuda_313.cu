 #include <stdio.h>
#include <cuda_runtime.h>

#define DIM 128
#define N 1024

// CUDA kernel
__global__ void solveLowerKernel(const double *lower, const double *b, double *buf, int dim, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < n) {
        for (int i = 0; i < dim; i++) {
            double val = b[k * dim + i];
            for (int j = 0; j < i; j++) {
                val -= lower[i * dim + j] * buf[k * dim + j];
            }
            buf[k * dim + i] = val / lower[i * dim + i];
        }
    }
}

int main() {
    // Allocate device memory
    double *d_lower, *d_b, *d_buf;

    cudaMalloc((void **)&d_lower, DIM * DIM * sizeof(double));
    cudaMalloc((void **)&d_b, N * DIM * sizeof(double));
    cudaMalloc((void **)&d_buf, N * DIM * sizeof(double));

    // Set grid and block sizes
    dim3 blockSize(256);  // You may adjust the block size
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    // Launch the kernel
    solveLowerKernel<<<gridSize, blockSize>>>(d_lower, d_b, d_buf, DIM, N);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Check for errors during the kernel launch
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    // Cleanup
    cudaFree(d_lower);
    cudaFree(d_b);
    cudaFree(d_buf);

    return 0;
}

