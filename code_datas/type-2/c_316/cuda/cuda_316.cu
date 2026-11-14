#include <stdio.h>
#include <cuda_runtime.h>

#define numARows 1024
#define numAColumns 1024
#define numBRows 1024
#define numBColumns 1024

// CUDA kernel
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int numCRows = numARows;
    int numCColumns = numBColumns;

    if (row < numCRows && col < numCColumns) {
        float sum = 0;

        for (int k = 0; k < numBRows; k++) {
            sum += A[row * numAColumns + k] * B[k * numBColumns + col];
        }

        C[row * numCColumns + col] = sum;
    }
}

int main() {
    // Allocate device memory
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, numARows * numAColumns * sizeof(float));
    cudaMalloc((void **)&d_B, numBRows * numBColumns * sizeof(float));
    cudaMalloc((void **)&d_C, numARows * numBColumns * sizeof(float));

    // Set grid and block sizes
    dim3 blockSize(16, 16);  // You may adjust the block size
    dim3 gridSize((numBColumns + blockSize.x - 1) / blockSize.x, (numARows + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, numARows, numAColumns, numBRows, numBColumns);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Check for errors during the kernel launch
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
 
