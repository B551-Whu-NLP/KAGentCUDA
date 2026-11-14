#include <stdio.h>

// CUDA kernel function
__global__ void addKernel(int* a, int* b, int* c, int vectorSize, int elements_per_thread) {
    int start = (blockIdx.x * blockDim.x + threadIdx.x) * elements_per_thread;

    for (int i = start; i - start < elements_per_thread && (i < vectorSize); i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Vector size and elements per thread
    int vectorSize = 1000;        // Change this according to your requirements
    int elements_per_thread = 10; // Change this according to your requirements

    // Host arrays
    int* h_a = (int*)malloc(vectorSize * sizeof(int));
    int* h_b = (int*)malloc(vectorSize * sizeof(int));
    int* h_c = (int*)malloc(vectorSize * sizeof(int));

    // Initialize host input arrays
    for (int i = 0; i < vectorSize; ++i) {
        h_a[i] = i; // Example data, you can modify this accordingly
        h_b[i] = i * 2;
    }

    // Device arrays
    int* d_a;
    int* d_b;
    int* d_c;
    cudaMalloc((void**)&d_a, vectorSize * sizeof(int));
    cudaMalloc((void**)&d_b, vectorSize * sizeof(int));
    cudaMalloc((void**)&d_c, vectorSize * sizeof(int));

    // Copy host input arrays to device
    cudaMemcpy(d_a, h_a, vectorSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, vectorSize * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((vectorSize + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    addKernel<<<grid_size, block_size>>>(d_a, d_b, d_c, vectorSize, elements_per_thread);

    // Copy the result back to the host
    cudaMemcpy(h_c, d_c, vectorSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Result: ");
    for (int i = 0; i < vectorSize; ++i) {
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
 
