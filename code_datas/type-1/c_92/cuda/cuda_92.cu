#include <stdio.h>

// CUDA kernel for divide and count
__global__ void devidecount(long Xsize, long Ysize, long Zsize, double* pint, int* pcount) {
    int n = Xsize * Ysize * 2 + (Ysize - 2) * Zsize * 2 + (Xsize - 2) * (Zsize - 2) * 2;
    long tid = threadIdx.x + blockDim.x * blockIdx.x;

    while (tid < n * n) {
        if (pcount[tid] > 1) {
            pint[tid] /= pcount[tid];
        }

        tid += blockDim.x * gridDim.x;
    }
}

int main() {
    // Set your desired parameters
    long Xsize = 512;
    long Ysize = 512;
    long Zsize = 512;

    // Allocate memory on the host
    double* h_pint = (double*)malloc(Xsize * Ysize * Zsize * sizeof(double));
    int* h_pcount = (int*)malloc(Xsize * Ysize * Zsize * sizeof(int));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    double* d_pint;
    int* d_pcount;
    cudaMalloc((void**)&d_pint, Xsize * Ysize * Zsize * sizeof(double));
    cudaMalloc((void**)&d_pcount, Xsize * Ysize * Zsize * sizeof(int));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((n + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);

    // Launch the CUDA kernel for divide and count
    devidecount<<<gridSize, blockSize>>>(Xsize, Ysize, Zsize, d_pint, d_pcount);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_pint);
    cudaFree(d_pcount);

    // Free host memory
    free(h_pint);
    free(h_pcount);

    return 0;
}
