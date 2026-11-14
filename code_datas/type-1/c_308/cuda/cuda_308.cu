#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void expandBoxes(const float *input, float *output, int dims, int clsNum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dims) {
        return;
    }

    int k = tid / clsNum;
    output[tid * 4 + 0] = input[k * 4 + 0];
    output[tid * 4 + 1] = input[k * 4 + 1];
    output[tid * 4 + 2] = input[k * 4 + 2];
    output[tid * 4 + 3] = input[k * 4 + 3];
}

int main() {
    // Set your problem dimensions
    const int dims = 256;   // Adjust as needed
    const int clsNum = 4;   // Adjust as needed

    // Allocate host memory
    float *h_input = (float *)malloc(dims * clsNum * sizeof(float));
    float *h_output = (float *)malloc(dims * 4 * sizeof(float));

    // Initialize host data (replace with your data initialization logic)
    for (int i = 0; i < dims * clsNum; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, dims * clsNum * sizeof(float));
    cudaMalloc((void **)&d_output, dims * 4 * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, dims * clsNum * sizeof(float), cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 blockSize(256); // You may adjust the block size
    dim3 gridSize((dims + blockSize.x - 1) / blockSize.x);

    // Launch the kernel
    expandBoxes<<<gridSize, blockSize>>>(d_input, d_output, dims, clsNum);

    // Copy result back to host (optional, depends on your application)
    cudaMemcpy(h_output, d_output, dims * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // Print or process the result as needed
    printf("Results printed here:\n");

    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
 
