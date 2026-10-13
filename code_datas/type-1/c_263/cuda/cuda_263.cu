#include <stdio.h>

// CUDA kernel function
__global__ void clip_kernel(int N, float ALPHA, float* X, int INCX, float* Y, int INCY) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < N) {
        float val = X[i * INCX];
        Y[i * INCY] = (val > ALPHA) ? val : 0;
    }
}

int main() {
    // Array size
    int N = 100; // Change this according to your requirements

    // Host arrays
    float* h_X = (float*)malloc(N * sizeof(float));
    float* h_Y = (float*)malloc(N * sizeof(float));

    // Initialize host input array
    for (int i = 0; i < N; ++i) {
        h_X[i] = i - N / 2.0; // Example data, you can modify this accordingly
    }

    // Device arrays
    float* d_X;
    float* d_Y;
    cudaMalloc((void**)&d_X, N * sizeof(float));
    cudaMalloc((void**)&d_Y, N * sizeof(float));

    // Copy host input array to device
    cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((N + block_size - 1) / block_size, 1);

    // Threshold value
    float ALPHA = 5.0; // Change this according to your requirements

    // Launch the CUDA kernel
    clip_kernel<<<grid_size, block_size>>>(N, ALPHA, d_X, 1, d_Y, 1);

    // Copy the result back to the host
    cudaMemcpy(h_Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Input Array:\n");
    for (int i = 0; i < N; ++i) {
        printf("%.2f ", h_X[i]);
    }

    printf("\nClipped Array (ALPHA=%.2f):\n", ALPHA);
    for (int i = 0; i < N; ++i) {
        printf("%.2f ", h_Y[i]);
    }
    printf("\n");

    // Clean up
    free(h_X);
    free(h_Y);
    cudaFree(d_X);
    cudaFree(d_Y);

    return 0;
}
 
