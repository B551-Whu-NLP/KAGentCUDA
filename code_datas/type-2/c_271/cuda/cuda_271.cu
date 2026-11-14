#include <stdio.h>

// CUDA kernel function
__global__ void mean(float* A, float* means, int size_row, int size_col) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size_col) {
        float sum = 0.0f;
        for (int i = 0; i < size_row; ++i) {
            sum += A[idx * size_row + i];
        }
        means[idx] = sum / size_row;
    }
}

int main() {
    // Array dimensions
    int size_row = 3; // Change this according to your requirements
    int size_col = 4; // Change this according to your requirements

    // Host arrays
    float* h_A = (float*)malloc(size_row * size_col * sizeof(float));
    float* h_means = (float*)malloc(size_col * sizeof(float));

    // Initialize host input array (A)
    for (int i = 0; i < size_row * size_col; ++i) {
        h_A[i] = i + 1; // Example data for A, you can modify this accordingly
    }

    // Device arrays
    float* d_A;
    float* d_means;
    cudaMalloc((void**)&d_A, size_row * size_col * sizeof(float));
    cudaMalloc((void**)&d_means, size_col * sizeof(float));

    // Copy host input array to device
    cudaMemcpy(d_A, h_A, size_row * size_col * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((size_col + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    mean<<<grid_size, block_size>>>(d_A, d_means, size_row, size_col);

    // Copy the result back to the host
    cudaMemcpy(h_means, d_means, size_col * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Array A:\n");
    for (int i = 0; i < size_row * size_col; ++i) {
        printf("%.2f ", h_A[i]);
    }

    printf("\nColumn Means:\n");
    for (int i = 0; i < size_col; ++i) {
        printf("Column %d: %.2f\n", i, h_means[i]);
    }

    // Clean up
    free(h_A);
    free(h_means);
    cudaFree(d_A);
    cudaFree(d_means);

    return 0;
}
 
