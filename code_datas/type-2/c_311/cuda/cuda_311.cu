#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Define matrix size (change as needed)
#define WIDTH 1024

// CUDA kernel
__global__ void matrixMult(float *A, float *B, float *C, int width) {
    int k = 0;
    float sum = 0;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < width) {
        for (k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

int main() {
    // Allocate host memory
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(WIDTH * WIDTH * sizeof(float));
    h_B = (float *)malloc(WIDTH * WIDTH * sizeof(float));
    h_C = (float *)malloc(WIDTH * WIDTH * sizeof(float));

    // Initialize host data (replace with your initialization logic)
    for (int i = 0; i < WIDTH * WIDTH; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, WIDTH * WIDTH * sizeof(float));
    cudaMalloc((void **)&d_B, WIDTH * WIDTH * sizeof(float));
    cudaMalloc((void **)&d_C, WIDTH * WIDTH * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, WIDTH * WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 blockSize(16, 16); // You may adjust the block size
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (WIDTH + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    matrixMult<<<gridSize, blockSize>>>(d_A, d_B, d_C, WIDTH);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, WIDTH * WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

    // Print or process the result as needed
    printf("Results printed here:\n");

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
 
