#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel
__global__ void normalize_kernel(int N, float *x, float *mean, float *variance, int batch, int filters, int spatial) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        int f = (index / spatial) % filters;
        x[index] = (x[index] - mean[f]) / sqrtf(variance[f] + 0.00001f);
    }
}

int main() {
    // Set your problem dimensions
    const int N = 1024;  // Set your actual problem size
    const int batch = 2; // Set your batch size
    const int filters = 3; // Set your number of filters
    const int spatial = 4; // Set your spatial dimension

    // Allocate host memory
    float *h_x = (float *)malloc(N * sizeof(float));
    float *h_mean = (float *)malloc(filters * sizeof(float));
    float *h_variance = (float *)malloc(filters * sizeof(float));

    // Initialize host data (replace with your data initialization logic)
    for (int i = 0; i < N; i++) {
        h_x[i] = (float)rand() / RAND_MAX;
    }

    for (int i = 0; i < filters; i++) {
        h_mean[i] = (float)rand() / RAND_MAX;
        h_variance[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory
    float *d_x, *d_mean, *d_variance;
    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_mean, filters * sizeof(float));
    cudaMalloc((void **)&d_variance, filters * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, h_mean, filters * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variance, h_variance, filters * sizeof(float), cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    // Launch the kernel
    normalize_kernel<<<gridSize, blockSize>>>(N, d_x, d_mean, d_variance, batch, filters, spatial);

    // Copy result back to host if needed
    cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print or process the result as needed
    printf("Normalized data:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", h_x[i]);
    }
    printf("\n");

    // Cleanup
    free(h_x);
    free(h_mean);
    free(h_variance);
    cudaFree(d_x);
    cudaFree(d_mean);
    cudaFree(d_variance);

    return 0;
}
