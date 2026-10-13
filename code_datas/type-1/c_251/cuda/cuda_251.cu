#include <stdio.h>
#include <math.h>

// CUDA kernel function
__global__ void clamp_kernel(int N, float* X, int INCX, float clamp_min, float clamp_max) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    
    if (i < N) {
        X[i * INCX] = fminf(clamp_max, fmaxf(clamp_min, X[i * INCX]));
    }
}

int main() {
    // Vector size
    int N = 1000; // Change this according to your requirements

    // Host array
    float* h_X = (float*)malloc(N * sizeof(float));

    // Initialize host input array
    for (int i = 0; i < N; ++i) {
        h_X[i] = static_cast<float>(i - 500); // Example data, you can modify this accordingly
    }

    // Device array
    float* d_X;
    cudaMalloc((void**)&d_X, N * sizeof(float));

    // Copy host input array to device
    cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((N + block_size - 1) / block_size, 1);

    // Set clamp values
    float clamp_min = -100.0f;
    float clamp_max = 100.0f;

    // Launch the CUDA kernel
    clamp_kernel<<<grid_size, block_size>>>(N, d_X, 1, clamp_min, clamp_max);

    // Copy the result back to the host
    cudaMemcpy(h_X, d_X, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Result: ");
    for (int i = 0; i < N; ++i) {
        printf("%f ", h_X[i]);
    }
    printf("\n");

    // Clean up
    free(h_X);
    cudaFree(d_X);

    return 0;
}
 
