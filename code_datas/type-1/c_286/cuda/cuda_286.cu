#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void smallCorrelation(float* L, float* innerSums, int innerSumsLength) {
    int u = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (u >= innerSumsLength)
        return;

    int realIdx = 2 * u;
    int imagIdx = realIdx + 1;
    L[u] = (innerSums[realIdx] * innerSums[realIdx]) + (innerSums[imagIdx] * innerSums[imagIdx]);
}

int main() {
    // Set the parameters
    const int innerSumsLength = 100; // Change this according to your requirements

    // Host arrays
    float* h_L = (float*)malloc(innerSumsLength * sizeof(float));
    float* h_innerSums = (float*)malloc(2 * innerSumsLength * sizeof(float)); // Assuming innerSums contains real and imaginary parts interleaved

    // Initialize host arrays (example data, modify as needed)
    for (int i = 0; i < 2 * innerSumsLength; ++i) {
        h_innerSums[i] = i; // Example data, you can modify this accordingly
    }

    // Device arrays
    float* d_L;
    float* d_innerSums;

    cudaMalloc((void**)&d_L, innerSumsLength * sizeof(float));
    cudaMalloc((void**)&d_innerSums, 2 * innerSumsLength * sizeof(float));

    // Copy host data to device
    cudaMemcpy(d_innerSums, h_innerSums, 2 * innerSumsLength * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int threadsPerBlock = 256; // Adjust this according to your requirements
    int blocksPerGrid = (innerSumsLength + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    smallCorrelation<<<blocksPerGrid, threadsPerBlock>>>(d_L, d_innerSums, innerSumsLength);

    // Copy the result back to the host (optional)
    cudaMemcpy(h_L, d_L, innerSumsLength * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Resultant array:\n");
    for (int i = 0; i < innerSumsLength; ++i) {
        printf("%.2f\t", h_L[i]);
    }
    printf("\n");

    // Clean up
    free(h_L);
    free(h_innerSums);
    cudaFree(d_L);
    cudaFree(d_innerSums);

    return 0;
}
 
