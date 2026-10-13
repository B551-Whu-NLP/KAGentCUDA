#include <stdio.h>

// CUDA kernel
__global__ void gpuMatrMultD(float* Ad, float* Bd, float* Cd, int rowsA, int colsA, int colsB) {
    int bIndx = blockIdx.x;
    int bIndy = blockIdx.y;
    int tIndx = threadIdx.x;
    int tIndy = threadIdx.y;

    Cd[(blockDim.x * bIndx + tIndx) * colsB + blockDim.y * bIndy + tIndy] = 0;

    for (int k = 0; k < colsA; ++k) {
        Cd[(blockDim.x * bIndx + tIndx) * colsB + blockDim.y * bIndy + tIndy] +=
            Ad[(blockDim.x * bIndx + tIndx) * colsA + k] * Bd[k * colsB + blockDim.y * bIndy + tIndy];
    }
}

int main() {
    // Your main program logic here

    // Example: Allocate and initialize host and device memory
    int rowsA = 512;    // Replace with your actual rowsA
    int colsA = 256;    // Replace with your actual colsA
    int colsB = 128;    // Replace with your actual colsB

    float* h_Ad = (float*)malloc(rowsA * colsA * sizeof(float));
    float* h_Bd = (float*)malloc(colsA * colsB * sizeof(float));
    float* h_Cd = (float*)malloc(rowsA * colsB * sizeof(float));

    float* d_Ad, * d_Bd, * d_Cd;
    cudaMalloc((void**)&d_Ad, rowsA * colsA * sizeof(float));
    cudaMalloc((void**)&d_Bd, colsA * colsB * sizeof(float));
    cudaMalloc((void**)&d_Cd, rowsA * colsB * sizeof(float));

    // Copy host memory to device
    cudaMemcpy(d_Ad, h_Ad, rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bd, h_Bd, colsA * colsB * sizeof(float), cudaMemcpyHostToDevice);

    // Configure and launch the CUDA kernel
    dim3 blockSize(16, 16); // Adjust block dimensions based on your requirements
    dim3 gridSize((colsB + blockSize.x - 1) / blockSize.x, (rowsA + blockSize.y - 1) / blockSize.y);

    gpuMatrMultD<<<gridSize, blockSize>>>(d_Ad, d_Bd, d_Cd, rowsA, colsA, colsB);

    // Copy device memory back to host
    cudaMemcpy(h_Cd, d_Cd, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);

    // Your post-kernel logic here

    // Free allocated memory
    free(h_Ad);
    free(h_Bd);
    free(h_Cd);
    cudaFree(d_Ad);
    cudaFree(d_Bd);
    cudaFree(d_Cd);

    return 0;
}
