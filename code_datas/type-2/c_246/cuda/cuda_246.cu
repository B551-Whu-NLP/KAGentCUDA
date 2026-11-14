#include <stdio.h>

// CUDA kernel function
__global__ void gpuSearchPosShmem1EQ(int key, int* devKey, int* devPos, int size) {
    int globalTx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalTx < size) {
        if (devKey[globalTx] == key) {
            devPos[0] = globalTx;
        }
    }
}

int main() {
    // Array size and search key
    int size = 10; // Change this according to your requirements
    int key = 5;   // Change this according to your requirements

    // Host arrays
    int* h_devKey = (int*)malloc(size * sizeof(int));
    int* h_devPos = (int*)malloc(sizeof(int));

    // Initialize host input array
    for (int i = 0; i < size; ++i) {
        h_devKey[i] = i; // Example data, you can modify this accordingly
    }

    // Device arrays
    int* d_devKey;
    int* d_devPos;
    cudaMalloc((void**)&d_devKey, size * sizeof(int));
    cudaMalloc((void**)&d_devPos, sizeof(int));

    // Copy host input array to device
    cudaMemcpy(d_devKey, h_devKey, size * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((size + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    gpuSearchPosShmem1EQ<<<grid_size, block_size>>>(key, d_devKey, d_devPos, size);

    // Copy the result back to the host
    cudaMemcpy(h_devPos, d_devPos, sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Position of key %d: %d\n", key, h_devPos[0]);

    // Clean up
    free(h_devKey);
    free(h_devPos);
    cudaFree(d_devKey);
    cudaFree(d_devPos);

    return 0;
}
 
