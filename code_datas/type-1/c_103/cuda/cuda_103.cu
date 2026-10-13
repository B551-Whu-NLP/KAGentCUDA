#include <stdio.h>

__global__ void kernelMaximum(float* maxhd, float* maxvd, int start, int size) {
    int tx = start + threadIdx.x;

    for (int i = size >> 1; i > 0; i >>= 1) {
        __syncthreads();

        if (tx < i) {
            if (maxhd[tx] < maxhd[tx + i]) maxhd[tx] = maxhd[tx + i];
            if (maxvd[tx] < maxvd[tx + i]) maxvd[tx] = maxvd[tx + i];
        }
    }
}

int main() {
    // Set your desired parameters
    int start = 0;    // Set your desired value for start
    int size = 1024;  // Set your desired value for size

    // Allocate memory on the host
    float* h_maxhd = (float*)malloc(size * sizeof(float));
    float* h_maxvd = (float*)malloc(size * sizeof(float));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    float* d_maxhd, * d_maxvd;
    cudaMalloc((void**)&d_maxhd, size * sizeof(float));
    cudaMalloc((void**)&d_maxvd, size * sizeof(float));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize(1, 1, 1);
    dim3 blockSize(size, 1, 1);

    // Launch the CUDA kernel for finding maximum values
    kernelMaximum<<<gridSize, blockSize>>>(d_maxhd, d_maxvd, start, size);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_maxhd);
    cudaFree(d_maxvd);

    // Free host memory
    free(h_maxhd);
    free(h_maxvd);

    return 0;
}
