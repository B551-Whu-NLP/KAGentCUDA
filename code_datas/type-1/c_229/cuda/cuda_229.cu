#include <stdio.h>

// CUDA kernel function
__global__ void transferMBR3(double* xy_copy, long long* a_copy, int tasks) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tasks; i += blockDim.x * gridDim.x) {
        a_copy[i] = static_cast<long long>(xy_copy[i] * 10000000);
    }
}

int main() {
    // Vector size
    int tasks = 100; // Change this according to your requirements

    // Host vectors
    double* h_xy_copy;
    long long* h_a_copy;
    h_xy_copy = (double*)malloc(tasks * sizeof(double));
    h_a_copy = (long long*)malloc(tasks * sizeof(long long));

    // Initialize host vectors
    for (int i = 0; i < tasks; ++i) {
        h_xy_copy[i] = i * 1.5; // Example data, you can modify this accordingly
    }

    // Device vectors
    double* d_xy_copy;
    long long* d_a_copy;
    cudaMalloc((void**)&d_xy_copy, tasks * sizeof(double));
    cudaMalloc((void**)&d_a_copy, tasks * sizeof(long long));

    // Copy host vectors to device
    cudaMemcpy(d_xy_copy, h_xy_copy, tasks * sizeof(double), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    int grid_size = (tasks + block_size - 1) / block_size;

    // Launch the CUDA kernel
    transferMBR3<<<grid_size, block_size>>>(d_xy_copy, d_a_copy, tasks);

    // Copy the result back to the host
    cudaMemcpy(h_a_copy, d_a_copy, tasks * sizeof(long long), cudaMemcpyDeviceToHost);

    // Display the result
    for (int i = 0; i < tasks; ++i) {
        printf("%lld ", h_a_copy[i]);
    }
    printf("\n");

    // Clean up
    free(h_xy_copy);
    free(h_a_copy);
    cudaFree(d_xy_copy);
    cudaFree(d_a_copy);

    return 0;
}
 
