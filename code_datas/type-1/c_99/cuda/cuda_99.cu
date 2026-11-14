#include <stdio.h>
#include <math.h>

// CUDA kernel for global calculation
__global__ void globalCalculateKernel(float* c, float* a, float* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    c[i * j] = sin(a[i * j]) * sin(a[i * j]) + cos(b[i * j]) * cos(b[i * j]) * cos(b[i * j]);
}

int main() {
    // Set your desired parameters
    int width = 512; // Set your desired value for width

    // Allocate memory on the host
    float* h_c = (float*)malloc(width * width * sizeof(float));
    float* h_a = (float*)malloc(width * width * sizeof(float));
    float* h_b = (float*)malloc(width * width * sizeof(float));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    float* d_c, * d_a, * d_b;
    cudaMalloc((void**)&d_c, width * width * sizeof(float));
    cudaMalloc((void**)&d_a, width * width * sizeof(float));
    cudaMalloc((void**)&d_b, width * width * sizeof(float));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((width + 15) / 16, (width + 15) / 16, 1);
    dim3 blockSize(16, 16, 1);

    // Launch the CUDA kernel for global calculation
    globalCalculateKernel<<<gridSize, blockSize>>>(d_c, d_a, d_b);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);

    // Free host memory
    free(h_c);
    free(h_a);
    free(h_b);

    return 0;
}
