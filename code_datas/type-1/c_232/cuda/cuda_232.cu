#include <stdio.h>

// CUDA kernel function
__global__ void memcpy_kernel(int* dst, int* src, size_t n) {
    int num = gridDim.x * blockDim.x;
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = id; i < n / sizeof(int); i += num) {
        dst[i] = src[i];
    }
}

int main() {
    // Vector size
    size_t n = 100; // Change this according to your requirements

    // Host vectors
    int* h_src;
    int* h_dst;
    h_src = (int*)malloc(n * sizeof(int));
    h_dst = (int*)malloc(n * sizeof(int));

    // Initialize host vectors
    for (size_t i = 0; i < n; ++i) {
        h_src[i] = i; // Example data, you can modify this accordingly
    }

    // Device vectors
    int* d_src;
    int* d_dst;
    cudaMalloc((void**)&d_src, n * sizeof(int));
    cudaMalloc((void**)&d_dst, n * sizeof(int));

    // Copy host vectors to device
    cudaMemcpy(d_src, h_src, n * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((n + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    memcpy_kernel<<<grid_size, block_size>>>(d_dst, d_src, n * sizeof(int));

    // Copy the result back to the host
    cudaMemcpy(h_dst, d_dst, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    for (size_t i = 0; i < n; ++i) {
        printf("%d ", h_dst[i]);
    }
    printf("\n");

    // Clean up
    free(h_src);
    free(h_dst);
    cudaFree(d_src);
    cudaFree(d_dst);

    return 0;
}
 
