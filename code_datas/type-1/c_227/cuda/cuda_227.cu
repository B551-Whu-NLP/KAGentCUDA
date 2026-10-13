#include <stdio.h>

// CUDA kernel function
__global__ void vadd(const float* a, const float* b, float* c, const unsigned int count) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < count) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Vector size
    unsigned int vector_size = 100; // Change this according to your requirements

    // Host vectors
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(vector_size * sizeof(float));
    h_b = (float*)malloc(vector_size * sizeof(float));
    h_c = (float*)malloc(vector_size * sizeof(float));

    // Initialize host vectors
    for (unsigned int i = 0; i < vector_size; ++i) {
        h_a[i] = i * 1.5f; // Example data, you can modify this accordingly
        h_b[i] = i * 2.0f; // Example data, you can modify this accordingly
    }

    // Device vectors
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, vector_size * sizeof(float));
    cudaMalloc((void**)&d_b, vector_size * sizeof(float));
    cudaMalloc((void**)&d_c, vector_size * sizeof(float));

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, vector_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, vector_size * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    int grid_size = (vector_size + block_size - 1) / block_size;

    // Launch the CUDA kernel
    vadd<<<grid_size, block_size>>>(d_a, d_b, d_c, vector_size);

    // Copy the result back to the host
    cudaMemcpy(h_c, d_c, vector_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    for (unsigned int i = 0; i < vector_size; ++i) {
        printf("%f ", h_c[i]);
    }
    printf("\n");

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
 
