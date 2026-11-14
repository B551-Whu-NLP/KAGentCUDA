#include <stdio.h>

// CUDA kernel for recursive reduction
__global__ void gpuReduceRecursive(int* I, int* O, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= n) return;

    int* N = I + blockIdx.x * blockDim.x;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0)
            N[tid] += N[tid + stride];

        __syncthreads();
    }

    if (tid == 0)
        O[blockIdx.x] = N[0];
}

int main() {
    // Set your desired parameters
    unsigned int n = 512;

    // Allocate memory on the host
    int* h_I = (int*)malloc(n * sizeof(int));
    int* h_O = (int*)malloc((n + 255) / 256 * sizeof(int));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    int* d_I, * d_O;
    cudaMalloc((void**)&d_I, n * sizeof(int));
    cudaMalloc((void**)&d_O, (n + 255) / 256 * sizeof(int));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((n + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);

    // Launch the CUDA kernel for recursive reduction
    gpuReduceRecursive<<<gridSize, blockSize>>>(d_I, d_O, n);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_I);
    cudaFree(d_O);

    // Free host memory
    free(h_I);
    free(h_O);

    return 0;
}
