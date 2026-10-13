#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void roundOff(float* mat, int N) {
    int i = threadIdx.x;
    int j = blockIdx.x;

    if (mat[i * N + j] >= 0)
        mat[i * N + j] = (int)(mat[i * N + j] + 0.5);
    else
        mat[i * N + j] = (int)(mat[i * N + j] - 0.5);
}

int main() {
    // Set the parameters
    const int N = 100; // Change this according to your requirements

    // Host array
    float* h_mat = (float*)malloc(N * N * sizeof(float));

    // Initialize host array (example data, modify as needed)
    for (int i = 0; i < N * N; ++i) {
        h_mat[i] = i; // Example data, you can modify this accordingly
    }

    // Device array
    float* d_mat;
    cudaMalloc((void**)&d_mat, N * N * sizeof(float));

    // Copy host data to device
    cudaMemcpy(d_mat, h_mat, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int threadsPerBlock = N;
    int blocksPerGrid = 1;

    // Launch the CUDA kernel
    roundOff<<<blocksPerGrid, threadsPerBlock>>>(d_mat, N);

    // Copy the result back to the host (optional)
    cudaMemcpy(h_mat, d_mat, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Result matrix:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%.1f\t", h_mat[i * N + j]);
        }
        printf("\n");
    }

    // Clean up
    free(h_mat);
    cudaFree(d_mat);

    return 0;
}
 
