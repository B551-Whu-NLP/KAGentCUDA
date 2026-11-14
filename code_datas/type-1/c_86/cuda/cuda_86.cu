#include <stdio.h>

// CUDA kernel for Leaky ReLU backward pass
__global__ void LreluBackward(float* srcDiff, float* dstDiff, float* srcData, int data_size, float alpha) {
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int i = 0; i < data_size; i += num_threads) {
        int index = i + thread_index;

        if (index < data_size) {
            dstDiff[index] = srcDiff[index] * ((srcData[index] > 0) + (srcData[index] <= 0) * alpha);
        }
    }
}

int main() {
    // Set your desired parameters
    int data_size = 512;
    float alpha = 0.01; // You can set your own value for alpha

    // Allocate memory on the host
    float* h_srcDiff = (float*)malloc(data_size * sizeof(float));
    float* h_dstDiff = (float*)malloc(data_size * sizeof(float));
    float* h_srcData = (float*)malloc(data_size * sizeof(float));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    float* d_srcDiff, * d_dstDiff, * d_srcData;
    cudaMalloc((void**)&d_srcDiff, data_size * sizeof(float));
    cudaMalloc((void**)&d_dstDiff, data_size * sizeof(float));
    cudaMalloc((void**)&d_srcData, data_size * sizeof(float));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((data_size + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);

    // Launch the CUDA kernel for Leaky ReLU backward pass
    LreluBackward<<<gridSize, blockSize>>>(d_srcDiff, d_dstDiff, d_srcData, data_size, alpha);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_srcDiff);
    cudaFree(d_dstDiff);
    cudaFree(d_srcData);

    // Free host memory
    free(h_srcDiff);
    free(h_dstDiff);
    free(h_srcData);

    return 0;
}
