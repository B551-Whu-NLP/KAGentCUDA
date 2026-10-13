#include <stdio.h>

// CUDA kernel function
__global__ void MMDSelfComputeWithSum(float* x_average, int size_x, float* distance_matrix) {
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;

    for (int i = block_id; i < size_x; i += gridDim.x) {
        for (int j = thread_id + i; j < size_x; j += blockDim.x) {
            distance_matrix[i * size_x + j] = x_average[i] * x_average[j];
        }
    }
}

int main() {
    // Matrix size
    int size_x = 1024;  // Change this according to your requirements

    // Host arrays
    float* h_x_average = (float*)malloc(size_x * sizeof(float));
    float* h_distance_matrix = (float*)malloc(size_x * size_x * sizeof(float));

    // Initialize host input arrays
    for (int i = 0; i < size_x; ++i) {
        h_x_average[i] = float(i);  // Example data for x_average, you can modify this accordingly
    }

    // Device arrays
    float* d_x_average;
    float* d_distance_matrix;
    cudaMalloc((void**)&d_x_average, size_x * sizeof(float));
    cudaMalloc((void**)&d_distance_matrix, size_x * size_x * sizeof(float));

    // Copy host input arrays to device
    cudaMemcpy(d_x_average, h_x_average, size_x * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    int grid_size = (size_x + block_size - 1) / block_size;

    // Launch the CUDA kernel
    MMDSelfComputeWithSum<<<grid_size, block_size>>>(d_x_average, size_x, d_distance_matrix);

    // Copy the result back to the host
    cudaMemcpy(h_distance_matrix, d_distance_matrix, size_x * size_x * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Results:\n");
    for (int i = 0; i < size_x; ++i) {
        for (int j = 0; j < size_x; ++j) {
            printf("%f ", h_distance_matrix[i * size_x + j]);
        }
        printf("\n");
    }

    // Clean up
    free(h_x_average);
    free(h_distance_matrix);
    cudaFree(d_x_average);
    cudaFree(d_distance_matrix);

    return 0;
}
 
