#include <stdio.h>
#include <cuda_runtime.h>

#define NX 128
#define NY 64
#define B 512

// CUDA kernel
__global__ void deinter_kernel(int NX, float *X, int NY, float *Y, int B, float *OUT) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < (NX + NY) * B) {
        int b = i / (NX + NY);
        int j = i % (NX + NY);

        if (j < NX) {
            if (X)
                X[b * NX + j] += OUT[i];
        } else {
            if (Y)
                Y[b * NY + j - NX] += OUT[i];
        }
    }
}

int main() {
    // Allocate device memory
    float *d_X, *d_Y, *d_OUT;

    cudaMalloc((void **)&d_X, B * NX * sizeof(float));
    cudaMalloc((void **)&d_Y, B * NY * sizeof(float));
    cudaMalloc((void **)&d_OUT, B * (NX + NY) * sizeof(float));

    // Set grid and block sizes
    dim3 blockSize(256);  // You may adjust the block size
    dim3 gridSize((B * (NX + NY) + blockSize.x - 1) / blockSize.x);

    // Launch the kernel
    deinter_kernel<<<gridSize, blockSize>>>(NX, d_X, NY, d_Y, B, d_OUT);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Check for errors during the kernel launch
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_OUT);

    return 0;
}
 
