#include <stdio.h>

// CUDA kernel for converting decisions to bits
__global__ void cudaConvertToBits(int* bit_decisions, unsigned short* bit_stream, int dec_size) {
    int dec_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    int bit_index = dec_index * 2;

    if (dec_index >= dec_size)
        return;

    int curr_decision = bit_decisions[dec_index];
    bit_stream[bit_index] = ((curr_decision & 2) >> 1);
    bit_stream[bit_index + 1] = (curr_decision & 1);
}

int main() {
    // Set your desired parameters
    int dec_size = 512;

    // Allocate memory on the host
    int* h_bit_decisions = (int*)malloc(dec_size * sizeof(int));
    unsigned short* h_bit_stream = (unsigned short*)malloc(dec_size * 2 * sizeof(unsigned short));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    int* d_bit_decisions;
    unsigned short* d_bit_stream;
    cudaMalloc((void**)&d_bit_decisions, dec_size * sizeof(int));
    cudaMalloc((void**)&d_bit_stream, dec_size * 2 * sizeof(unsigned short));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((dec_size + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);

    // Launch the CUDA kernel for converting decisions to bits
    cudaConvertToBits<<<gridSize, blockSize>>>(d_bit_decisions, d_bit_stream, dec_size);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_bit_decisions);
    cudaFree(d_bit_stream);

    // Free host memory
    free(h_bit_decisions);
    free(h_bit_stream);

    return 0;
}
