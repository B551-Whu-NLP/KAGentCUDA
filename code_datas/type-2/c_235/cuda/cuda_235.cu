#include <stdio.h>

// CUDA kernel function
__global__ void find_max_among_blocks(int* data, int blockSize, int nbBlocks) {
    for (int i = 0; i < nbBlocks; ++i) {
        if (data[0] < data[i * blockSize]) {
            data[0] = data[i * blockSize];
        }
    }
}

int main() {
    // Vector size
    int dataSize = 100; // Change this according to your requirements

    // Number of blocks and block size
    int nbBlocks = 10;
    int blockSize = dataSize / nbBlocks;

    // Host vector
    int* h_data;
    h_data = (int*)malloc(dataSize * sizeof(int));

    // Initialize host vector
    for (int i = 0; i < dataSize; ++i) {
        h_data[i] = i; // Example data, you can modify this accordingly
    }

    // Device vector
    int* d_data;
    cudaMalloc((void**)&d_data, dataSize * sizeof(int));

    // Copy host vector to device
    cudaMemcpy(d_data, h_data, dataSize * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel
    find_max_among_blocks<<<1, 1>>>(d_data, blockSize, nbBlocks);

    // Copy the result back to the host
    cudaMemcpy(h_data, d_data, dataSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Maximum value among blocks: %d\n", h_data[0]);

    // Clean up
    free(h_data);
    cudaFree(d_data);

    return 0;
}
 
