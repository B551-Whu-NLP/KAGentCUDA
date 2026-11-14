#include <stdio.h>

// CUDA kernel function
__global__ void addVectorsInto(float* result, float* a, float* b, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        result[i] = a[i] + b[i];
    }
}

int main() {
    // Vector size
    int N = 100; // Change this according to your requirements

    // Host vectors
    float* h_result = (float*)malloc(N * sizeof(float));
    float* h_a = (float*)malloc(N * sizeof(float));
    float* h_b = (float*)malloc(N * sizeof(float));

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_a[i] = i + 1; // Example data, you can modify this accordingly
        h_b[i] = 2 * (i + 1);
    }

    // Device vectors
    float* d_result;
    float* d_a;
    float* d_b;
    cudaMalloc((void**)&d_result, N * sizeof(float));
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((N + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    addVectorsInto<<<grid_size, block_size>>>(d_result, d_a, d_b, N);

    // Copy the result back to the host
    cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Result: ");
    for (int i = 0; i < N; ++i) {
        printf("%f ", h_result[i]);
    }
    printf("\n");

    // Clean up
    free(h_result);
    free(h_a);
    free(h_b);
    cudaFree(d_result);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
 
