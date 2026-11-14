#include <stdio.h>

// CUDA kernel
__global__ void Kernel_Dot_reduction2(float* dev_c, float* reduction, int r, const int c, const int n, int size_block) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= r || j >= c) return;

    float temp = 0;
    for (int k = 0; k < size_block; k++) {
        temp += reduction[i * (c * size_block) + j * size_block + k];
    }

    dev_c[i * c + j] = temp;
}

int main() {
    // Your main program logic here

    // Example: Allocate and initialize host and device memory
    int r = 100; // Replace with your actual dimensions
    int c = 100;
    int n = 100;
    int size_block = 16;

    float* h_reduction = (float*)malloc(r * c * n * sizeof(float));
    float* h_dev_c = (float*)malloc(r * c * sizeof(float));

    float* d_reduction, * d_dev_c;
    cudaMalloc((void**)&d_reduction, r * c * n * sizeof(float));
    cudaMalloc((void**)&d_dev_c, r * c * sizeof(float));

    // Copy host memory to device
    cudaMemcpy(d_reduction, h_reduction, r * c * n * sizeof(float), cudaMemcpyHostToDevice);

    // Configure and launch the CUDA kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((r + blockSize.x - 1) / blockSize.x, (c + blockSize.y - 1) / blockSize.y);
    Kernel_Dot_reduction2<<<gridSize, blockSize>>>(d_dev_c, d_reduction, r, c, n, size_block);

    // Copy device memory back to host
    cudaMemcpy(h_dev_c, d_dev_c, r * c * sizeof(float), cudaMemcpyDeviceToHost);

    // Your post-kernel logic here

    // Free allocated memory
    free(h_reduction);
    free(h_dev_c);
    cudaFree(d_reduction);
    cudaFree(d_dev_c);

    return 0;
}
