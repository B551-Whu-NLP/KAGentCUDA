#include <stdio.h>

// CUDA kernel for Leaky ReLU forward pass
__global__ void LreluForward(float* srcData, float* dstData, int data_size, float alpha) {
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int i = 0; i < data_size; i += num_threads) {
        int index = i + thread_index;

        if (index < data_size) {
            dstData[index] = (srcData[index] > 0) ? srcData[index] : srcData[index] * alpha;
        }
    }
}

int main() {
    // Set your desired parameters
    int data_size = 512;
    float alpha = 0.01; // You can set your own value for alpha

    // Allocate memory on the host
    float* h_srcData = (float*)malloc(data_size * sizeof(float));
    float* h_dstData = (float*)malloc(data_size * sizeof(float));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    float* d_srcData, * d_dstData;
    cudaMalloc((void**)&d_srcData, data_size * sizeof(float));
    cudaMalloc((void**)&d_dstData, data_size * sizeof(float));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((data_size + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);

    // Launch the CUDA kernel for Leaky ReLU forward pass
    LreluForward<<<gridSize, blockSize>>>(d_srcData, d_dstData, data_size, alpha);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_srcData);
    cudaFree(d_dstData);

    // Free host memory
    free(h_srcData);
    free(h_dstData);

    return 0;
}
