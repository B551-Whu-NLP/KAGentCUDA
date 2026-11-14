#include <stdio.h>

// CUDA kernel for inner divide and count
__global__ void devidecountInner(long Xsize, long Ysize, long Zsize, double* p, double* pn, int* pcountinner) {
    long tid = threadIdx.x + blockDim.x * blockIdx.x;

    while (tid < Xsize * Ysize * Zsize) {
        if (pcountinner[tid] > 1) {
            p[tid] = pn[tid] / pcountinner[tid];
            pn[tid] = 0;
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
    double* h_p = (double*)malloc(Xsize * Ysize * Zsize * sizeof(double));
    double* h_pn = (double*)malloc(Xsize * Ysize * Zsize * sizeof(double));
    int* h_pcountinner = (int*)malloc(Xsize * Ysize * Zsize * sizeof(int));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    double* d_p, * d_pn;
    int* d_pcountinner;
    cudaMalloc((void**)&d_p, Xsize * Ysize * Zsize * sizeof(double));
    cudaMalloc((void**)&d_pn, Xsize * Ysize * Zsize * sizeof(double));
    cudaMalloc((void**)&d_pcountinner, Xsize * Ysize * Zsize * sizeof(int));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((Xsize * Ysize * Zsize + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);

    // Launch the CUDA kernel for inner divide and count
    devidecountInner<<<gridSize, blockSize>>>(Xsize, Ysize, Zsize, d_p, d_pn, d_pcountinner);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_p);
    cudaFree(d_pn);
    cudaFree(d_pcountinner);

    // Free host memory
    free(h_p);
    free(h_pn);
    free(h_pcountinner);

    return 0;
}
