#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel
__global__ void inter_kernel(int NX, float *X, int NY, float *Y, int B, float *OUT) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < (NX + NY) * B) {
        int b = i / (NX + NY);
        int j = i % (NX + NY);

        if (j < NX) {
            OUT[i] = X[b * NX + j];
        } else {
            OUT[i] = Y[b * NY + j - NX];
        }
    }
}

int main() {
    // Set your problem dimensions
    const int NX = 128;
    const int NY = 64;
    const int B = 256;

    // Allocate host memory
    float *h_X = (float *)malloc(NX * B * sizeof(float));
    float *h_Y = (float *)malloc(NY * B * sizeof(float));
    float *h_OUT = (float *)malloc((NX + NY) * B * sizeof(float));

    // Initialize host data (replace with your data initialization logic)
    for (int i = 0; i < NX * B; ++i) {
        h_X[i] = static_cast<float>(i);
    }

    for (int i = 0; i < NY * B; ++i) {
        h_Y[i] = static_cast<float>(i + NX * B);
    }

    // Allocate device memory
    float *d_X, *d_Y, *d_OUT;
    cudaMalloc((void **)&d_X, NX * B * sizeof(float));
    cudaMalloc((void **)&d_Y, NY * B * sizeof(float));
    cudaMalloc((void **)&d_OUT, (NX + NY) * B * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_X, h_X, NX * B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, NY * B * sizeof(float), cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 blockSize(256);  // You may adjust the block size
    dim3 gridSize((B * (NX + NY) + blockSize.x - 1) / blockSize.x);

    // Launch the kernel
    inter_kernel<<<gridSize, blockSize>>>(NX, d_X, NY, d_Y, B, d_OUT);

    // Copy result back to host
    cudaMemcpy(h_OUT, d_OUT, (NX + NY) * B * sizeof(float), cudaMemcpyDeviceToHost);

    // Print or process the result as needed
    printf("Results printed here:\n");

    // Cleanup
    free(h_X);
    free(h_Y);
    free(h_OUT);
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_OUT);

    return 0;
}
 
