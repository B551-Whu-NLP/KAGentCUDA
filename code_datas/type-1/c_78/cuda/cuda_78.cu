#include <stdio.h>

// CUDA kernel for resized class score
__global__ void resizedClsScore(const float* score, const float* score_factors, float* output, int dims) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= dims) {
        return;
    }

    if (score[tid] == (-1)) {
        output[tid] = -1;
    } else {
        output[tid] = score[tid] * score_factors[tid];
    }
}

int main() {
    // Set your desired dimensions
    int dims = 512;

    // Allocate memory on the host
    float* h_score = (float*)malloc(dims * sizeof(float));
    float* h_score_factors = (float*)malloc(dims * sizeof(float));
    float* h_output = (float*)malloc(dims * sizeof(float));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    float* d_score, * d_score_factors, * d_output;
    cudaMalloc((void**)&d_score, dims * sizeof(float));
    cudaMalloc((void**)&d_score_factors, dims * sizeof(float));
    cudaMalloc((void**)&d_output, dims * sizeof(float));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((dims + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);

    // Launch the CUDA kernel for resized class score
    resizedClsScore<<<gridSize, blockSize>>>(d_score, d_score_factors, d_output, dims);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_score);
    cudaFree(d_score_factors);
    cudaFree(d_output);

    // Free host memory
    free(h_score);
    free(h_score_factors);
    free(h_output);

    return 0;
}
