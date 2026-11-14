#include <stdio.h>

// CUDA kernel function
__global__ void axpy_kernel(int N, float ALPHA, float* X, int OFFX, int INCX, float* Y, int OFFY, int INCY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        Y[OFFY + i * INCY] += ALPHA * X[OFFX + i * INCX];
    }
}

int main() {
    // Vector size
    int N = 1000; // Change this according to your requirements

    // Host arrays
    float ALPHA = 2.0f;
    float* h_X = (float*)malloc(N * sizeof(float));
    float* h_Y = (float*)malloc(N * sizeof(float));

    // Initialize host input arrays
    for (int i = 0; i < N; ++i) {
        h_X[i] = i; // Example data, you can modify this accordingly
        h_Y[i] = i * 2;
    }

    // Device arrays
    float* d_X;
    float* d_Y;
    cudaMalloc((void**)&d_X, N * sizeof(float));
    cudaMalloc((void**)&d_Y, N * sizeof(float));

    // Copy host input arrays to device
    cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((N + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    axpy_kernel<<<grid_size, block_size>>>(N, ALPHA, d_X, 0, 1, d_Y, 0, 1);

    // Copy the result back to the host
    cudaMemcpy(h_Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Result: ");
    for (int i = 0; i < N; ++i) {
        printf("%f ", h_Y[i]);
    }
    printf("\n");

    // Clean up
    free(h_X);
    free(h_Y);
    cudaFree(d_X);
    cudaFree(d_Y);

    return 0;
}
 
