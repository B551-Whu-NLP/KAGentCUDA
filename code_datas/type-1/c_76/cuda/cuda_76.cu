#include <stdio.h>

// CUDA kernel for setting source positions
__global__ void cuda_set_sg(int* sxz, int sxbeg, int szbeg, int jsx, int jsz, int ns, int npml, int nnz) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    if (id < ns) {
        sxz[id] = nnz * (sxbeg + id * jsx + npml) + (szbeg + id * jsz + npml);
    }
}

int main() {
    // Set your desired parameters
    int ns = 512;  // Number of sources
    int npml = 10;  // PML size
    int nnz = 100;  // Some constant value
    int sxbeg = 0;
    int szbeg = 0;
    int jsx = 1;
    int jsz = 1;

    // Allocate memory on the host
    int* h_sxz = (int*)malloc(ns * sizeof(int));

    // Allocate memory on the device
    int* d_sxz;
    cudaMalloc((void**)&d_sxz, ns * sizeof(int));

    // Calculate grid and block dimensions
    dim3 gridSize((ns + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);

    // Launch the CUDA kernel for setting source positions
    cuda_set_sg<<<gridSize, blockSize>>>(d_sxz, sxbeg, szbeg, jsx, jsz, ns, npml, nnz);

    // Copy the result back to the host
    cudaMemcpy(h_sxz, d_sxz, ns * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_sxz);

    // Free host memory
    free(h_sxz);

    return 0;
}
