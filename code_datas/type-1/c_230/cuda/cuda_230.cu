#include <stdio.h>

// CUDA kernel function
__global__ void binarize_kernel(float* x, int n, float* binary) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    binary[i] = (x[i] >= 0) ? 1 : -1;
}

int main() {
    // Vector size
    int n = 100; // Change this according to your requirements

    // Host vectors
    float* h_x;
    float* h_binary;
    h_x = (float*)malloc(n * sizeof(float));
    h_binary = (float*)malloc(n * sizeof(float));

    // Initialize host vectors
    for (int i = 0; i < n; ++i) {
        h_x[i] = i - 50; // Example data, you can modify this accordingly
    }

    // Device vectors
    float* d_x;
    float* d_binary;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_binary, n * sizeof(float));

    // Copy host vectors to device
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((n + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    binarize_kernel<<<grid_size, block_size>>>(d_x, n, d_binary);

    // Copy the result back to the host
    cudaMemcpy(h_binary, d_binary, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    for (int i = 0; i < n; ++i) {
        printf("%f ", h_binary[i]);
    }
    printf("\n");

    // Clean up
    free(h_x);
    free(h_binary);
    cudaFree(d_x);
    cudaFree(d_binary);

    return 0;
}
 
