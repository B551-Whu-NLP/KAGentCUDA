#include <stdio.h>

// CUDA kernel function for subtraction
__global__ void intSubtract(int* c, const int* a, const int* b, const unsigned int d) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < d) {
        c[i] = a[i] - b[i];
    }
}

int main() {
    // Vector size
    unsigned int vector_size = 100; // Change this according to your requirements

    // Host vectors
    int *h_a, *h_b, *h_c;
    h_a = (int*)malloc(vector_size * sizeof(int));
    h_b = (int*)malloc(vector_size * sizeof(int));
    h_c = (int*)malloc(vector_size * sizeof(int));

    // Initialize host vectors
    for (unsigned int i = 0; i < vector_size; ++i) {
        h_a[i] = i * 2; // Example data, you can modify this accordingly
        h_b[i] = i;     // Example data, you can modify this accordingly
    }

    // Device vectors
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, vector_size * sizeof(int));
    cudaMalloc((void**)&d_b, vector_size * sizeof(int));
    cudaMalloc((void**)&d_c, vector_size * sizeof(int));

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, vector_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, vector_size * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    int grid_size = (vector_size + block_size - 1) / block_size;

    // Launch the CUDA kernel for subtraction
    intSubtract<<<grid_size, block_size>>>(d_c, d_a, d_b, vector_size);

    // Copy the result back to the host
    cudaMemcpy(h_c, d_c, vector_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    for (unsigned int i = 0; i < vector_size; ++i) {
        printf("%d ", h_c[i]);
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
 
