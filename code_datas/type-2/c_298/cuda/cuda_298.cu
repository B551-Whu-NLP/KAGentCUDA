#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel
__global__ void kernel_CEE(float *x, int *t, float *loss, int r, int c) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int N = r;
    float temp;

    while (i < N) {
        for (int j = 0; j < c; j++) {
            if (t[i * c + j] == 1) {
                temp = logf(x[i * c + j] + 1e-7);
                atomicAdd(loss, temp);
                continue;
            }
        }

        i += gridDim.x * blockDim.x;
    }
}

int main() {
    // Set your problem dimensions
    const int r = 100;  // Set your actual number of rows
    const int c = 10;   // Set your actual number of columns

    // Allocate host memory
    float *h_x = (float *)malloc(r * c * sizeof(float));
    int *h_t = (int *)malloc(r * c * sizeof(int));
    float *h_loss = (float *)malloc(sizeof(float));

    // Initialize host data (replace with your data initialization logic)
    for (int i = 0; i < r * c; i++) {
        h_x[i] = (float)rand() / RAND_MAX;
        h_t[i] = rand() % 2;  // Assuming binary labels (0 or 1)
    }

    // Allocate device memory
    float *d_x;
    int *d_t;
    float *d_loss;
    cudaMalloc((void **)&d_x, r * c * sizeof(float));
    cudaMalloc((void **)&d_t, r * c * sizeof(int));
    cudaMalloc((void **)&d_loss, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_x, h_x, r * c * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_t, h_t, r * c * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_loss, 0, sizeof(float));  // Initialize loss on device

    // Set grid and block sizes
    dim3 blockSize(256);  // You may adjust the block size
    dim3 gridSize((r + blockSize.x - 1) / blockSize.x);

    // Launch the kernel
    kernel_CEE<<<gridSize, blockSize>>>(d_x, d_t, d_loss, r, c);

    // Copy result back to host
    cudaMemcpy(h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    // Print or process the result as needed
    printf("Cross-Entropy Loss: %f\n", *h_loss);

    // Cleanup
    free(h_x);
    free(h_t);
    free(h_loss);
    cudaFree(d_x);
    cudaFree(d_t);
    cudaFree(d_loss);

    return 0;
}
