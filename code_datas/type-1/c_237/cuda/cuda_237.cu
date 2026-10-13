#include <stdio.h>

// CUDA kernel function
__global__ void expandScoreFactors(const float* input, float* output, int dims, int clsNum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dims) {
        return;
    }

    int k = tid / clsNum;
    output[tid] = input[k];
}

int main() {
    // Dimensions and class number
    int dims = 15;
    int clsNum = 3;

    // Host arrays
    float* h_input = (float*)malloc(clsNum * sizeof(float));
    float* h_output = (float*)malloc(dims * sizeof(float));

    // Initialize host input array
    for (int i = 0; i < clsNum; ++i) {
        h_input[i] = static_cast<float>(i + 1);  // Example data, you can modify this accordingly
    }

    // Device arrays
    float* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input, clsNum * sizeof(float));
    cudaMalloc((void**)&d_output, dims * sizeof(float));

    // Copy host input array to device
    cudaMemcpy(d_input, h_input, clsNum * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((dims + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    expandScoreFactors<<<grid_size, block_size>>>(d_input, d_output, dims, clsNum);

    // Copy the result back to the host
    cudaMemcpy(h_output, d_output, dims * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Expanded Output: ");
    for (int i = 0; i < dims; ++i) {
        printf("%f ", h_output[i]);
    }
    printf("\n");

    // Clean up
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
 
