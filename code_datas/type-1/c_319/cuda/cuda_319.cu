#include <stdio.h>
#include <cuda_runtime.h>

__global__ void boxesScale(const float *input, float *output, int dims, float scale0, float scale1, float scale2, float scale3) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dims) {
        return;
    }

    output[tid * 4] = input[tid * 4] / scale0;
    output[tid * 4 + 1] = input[tid * 4 + 1] / scale1;
    output[tid * 4 + 2] = input[tid * 4 + 2] / scale2;
    output[tid * 4 + 3] = input[tid * 4 + 3] / scale3;
}

int main() {
    // Set the dimensions and scales
    int dims = 128;         // Set the appropriate value
    float scale0 = 2.0f;    // Set the appropriate value
    float scale1 = 1.5f;    // Set the appropriate value
    float scale2 = 1.0f;    // Set the appropriate value
    float scale3 = 0.5f;    // Set the appropriate value

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, dims * 4 * sizeof(float));
    cudaMalloc((void **)&d_output, dims * 4 * sizeof(float));

    // Set grid and block sizes
    dim3 blockSize(256);   // You may adjust the block size
    dim3 gridSize((dims + blockSize.x - 1) / blockSize.x, 1);

    // Launch the kernel
    boxesScale<<<gridSize, blockSize>>>(d_input, d_output, dims, scale0, scale1, scale2, scale3);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Check for errors during the kernel launch
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
 
