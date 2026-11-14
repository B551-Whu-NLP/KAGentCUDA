#include <stdio.h>

// CUDA kernel for converting edge mask to float
__global__ void convertEdgeMaskToFloatDevice(float* d_output, unsigned char* d_input, unsigned int width, unsigned int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    d_output[y * width + x] = min(d_input[y * width + x], d_input[width * height + y * width + x]);
}

int main() {
    // Set your desired image dimensions
    unsigned int width = 512;
    unsigned int height = 512;

    // Allocate memory on the host
    float* h_output = (float*)malloc(width * height * sizeof(float));
    unsigned char* h_input = (unsigned char*)malloc(2 * width * height * sizeof(unsigned char));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    float* d_output;
    unsigned char* d_input;
    cudaMalloc((void**)&d_output, width * height * sizeof(float));
    cudaMalloc((void**)&d_input, 2 * width * height * sizeof(unsigned char));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    dim3 blockSize(16, 16);

    // Launch the CUDA kernel for converting edge mask to float
    convertEdgeMaskToFloatDevice<<<gridSize, blockSize>>>(d_output, d_input, width, height);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_output);
    cudaFree(d_input);

    // Free host memory
    free(h_output);
    free(h_input);

    return 0;
}
