#include <stdio.h>

// CUDA kernel function
__global__ void weighted_sum_kernel(int n, float* a, float* b, float* s, float* c) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = s[i] * a[i] + (1 - s[i]) * (b ? b[i] : 0);
    }
}

int main() {
    // Array size
    int size = 1024;  // Change this according to your requirements

    // Host arrays
    float* h_a = (float*)malloc(size * sizeof(float));
    float* h_b = (float*)malloc(size * sizeof(float));
    float* h_s = (float*)malloc(size * sizeof(float));
    float* h_c = (float*)malloc(size * sizeof(float));

    // Initialize host input arrays
    for (int i = 0; i < size; ++i) {
        h_a[i] = 1.0f;  // Example data for a, you can modify this accordingly
        h_b[i] = 2.0f;  // Example data for b, you can modify this accordingly
        h_s[i] = 0.5f;  // Example data for s, you can modify this accordingly
    }

    // Device arrays
    float* d_a;
    float* d_b;
    float* d_s;
    float* d_c;
    cudaMalloc((void**)&d_a, size * sizeof(float));
    cudaMalloc((void**)&d_b, size * sizeof(float));
    cudaMalloc((void**)&d_s, size * sizeof(float));
    cudaMalloc((void**)&d_c, size * sizeof(float));

    // Copy host input arrays to device
    cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, h_s, size * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    // Launch the CUDA kernel
    weighted_sum_kernel<<<grid_size, block_size>>>(size, d_a, d_b, d_s, d_c);

    // Copy the result back to the host
    cudaMemcpy(h_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Results:\n");
    for (int i = 0; i < size; ++i) {
        printf("%f ", h_c[i]);
    }
    printf("\n");

    // Clean up
    free(h_a);
    free(h_b);
    free(h_s);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_s);
    cudaFree(d_c);

    return 0;
}
 
