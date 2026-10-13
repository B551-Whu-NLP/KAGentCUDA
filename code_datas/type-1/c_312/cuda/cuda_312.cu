#include <stdio.h>
#include <cuda_runtime.h>

// Define dimensions (change as needed)
#define WIDTH 1024
#define HEIGHT 1024
#define DEPTH 1024

// CUDA kernel
__global__ void SetToZero_kernel(float *d_vx, float *d_vy, float *d_vz, int w, int h, int l) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = j * w + i;

    if (i < w && j < h) {
        for (int k = 0; k < l; ++k, index += w * h) {
            d_vx[index] = 0;
            d_vy[index] = 0;
            d_vz[index] = 0;
        }
    }
}

int main() {
    // Allocate device memory
    float *d_vx, *d_vy, *d_vz;
    cudaMalloc((void **)&d_vx, WIDTH * HEIGHT * DEPTH * sizeof(float));
    cudaMalloc((void **)&d_vy, WIDTH * HEIGHT * DEPTH * sizeof(float));
    cudaMalloc((void **)&d_vz, WIDTH * HEIGHT * DEPTH * sizeof(float));

    // Set grid and block sizes
    dim3 blockSize(16, 16);  // You may adjust the block size
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    SetToZero_kernel<<<gridSize, blockSize>>>(d_vx, d_vy, d_vz, WIDTH, HEIGHT, DEPTH);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Check for errors during the kernel launch
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    // Cleanup
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);

    return 0;
}
 
