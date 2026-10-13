#include <stdio.h>

// CUDA kernel function
__global__ void add_kernel(int N, float ALPHA, float* X, int INCX) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N)
        X[i * INCX] += ALPHA;
}

int main() {
    // Vector size
    int N = 100; // Change this according to your requirements

    // Host vector
    float* h_X;
    h_X = (float*)malloc(N * sizeof(float));

    // Initialize host vector
    for (int i = 0; i < N; ++i) {
        h_X[i] = i; // Example data, you can modify this accordingly
    }

    // Device vector
    float* d_X;
    cudaMalloc((void**)&d_X, N * sizeof(float));

    // Copy host vector to device
    cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((N + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    add_kernel<<<grid_size, block_size>>>(N, 2.0f, d_X, 1); // Example ALPHA is set to 2.0

    // Copy the result back to the host
    cudaMemcpy(h_X, d_X, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    for (int i = 0; i < N; ++i) {
        printf("%f ", h_X[i]);
    }
    printf("\n");

    // Clean up
    free(h_X);
    cudaFree(d_X);

    return 0;
}
 
