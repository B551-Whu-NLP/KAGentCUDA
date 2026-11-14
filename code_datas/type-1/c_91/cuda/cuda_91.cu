#include <stdio.h>

// CUDA kernel for circularity calculation
__global__ void circularity(const int compCount, const int* areaRes, const float* perimeterRes, float* circ) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < compCount) {
        circ[tid] = (4.0 * 3.14159265359 * (float)areaRes[tid]) / (perimeterRes[tid] * perimeterRes[tid]);
    }
}

int main() {
    // Set your desired parameters
    int compCount = 512;

    // Allocate memory on the host
    int* h_areaRes = (int*)malloc(compCount * sizeof(int));
    float* h_perimeterRes = (float*)malloc(compCount * sizeof(float));
    float* h_circ = (float*)malloc(compCount * sizeof(float));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    int* d_areaRes;
    float* d_perimeterRes;
    float* d_circ;
    cudaMalloc((void**)&d_areaRes, compCount * sizeof(int));
    cudaMalloc((void**)&d_perimeterRes, compCount * sizeof(float));
    cudaMalloc((void**)&d_circ, compCount * sizeof(float));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((compCount + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);

    // Launch the CUDA kernel for circularity calculation
    circularity<<<gridSize, blockSize>>>(compCount, d_areaRes, d_perimeterRes, d_circ);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_areaRes);
    cudaFree(d_perimeterRes);
    cudaFree(d_circ);

    // Free host memory
    free(h_areaRes);
    free(h_perimeterRes);
    free(h_circ);

    return 0;
}
