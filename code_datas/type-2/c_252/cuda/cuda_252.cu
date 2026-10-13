#include <stdio.h>

// CUDA kernel function
__global__ void histogram(int* x, int* bins, int n) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n) {
        const auto c = x[i];
        atomicAdd(&bins[c], 1);
    }
}

int main() {
    // Data size
    int n = 1000; // Change this according to your requirements

    // Host arrays
    int* h_x = (int*)malloc(n * sizeof(int));
    int* h_bins = (int*)calloc(n, sizeof(int)); // Initialize bins to zero

    // Initialize host input array
    for (int i = 0; i < n; ++i) {
        h_x[i] = i % 10; // Example data, you can modify this accordingly
    }

    // Device arrays
    int* d_x;
    int* d_bins;
    cudaMalloc((void**)&d_x, n * sizeof(int));
    cudaMalloc((void**)&d_bins, n * sizeof(int));

    // Copy host input array to device
    cudaMemcpy(d_x, h_x, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bins, h_bins, n * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((n + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    histogram<<<grid_size, block_size>>>(d_x, d_bins, n);

    // Copy the result back to the host
    cudaMemcpy(h_bins, d_bins, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Histogram Result:\n");
    for (int i = 0; i < n; ++i) {
        printf("Bin %d: %d\n", i, h_bins[i]);
    }

    // Clean up
    free(h_x);
    free(h_bins);
    cudaFree(d_x);
    cudaFree(d_bins);

    return 0;
}
 
