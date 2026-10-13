#include <stdio.h>

// CUDA kernel for grayscale conversion
__global__ void grayscale(unsigned char* input, unsigned char* output, int size) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < size) {
        unsigned char r, g, b;
        r = input[3 * i];
        g = input[3 * i + 1];
        b = input[3 * i + 2];

        output[i] = (unsigned char)(0.21 * (float)r + 0.71 * (float)g + 0.07 * (float)b);
    }
}

int main() {
    // Set your desired parameters
    int size = 512; // Set your desired value for size

    // Allocate memory on the host
    unsigned char* h_input = (unsigned char*)malloc(3 * size * sizeof(unsigned char));
    unsigned char* h_output = (unsigned char*)malloc(size * sizeof(unsigned char));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    unsigned char* d_input, * d_output;
    cudaMalloc((void**)&d_input, 3 * size * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, size * sizeof(unsigned char));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((size + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);

    // Launch the CUDA kernel for grayscale conversion
    grayscale<<<gridSize, blockSize>>>(d_input, d_output, size);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}
