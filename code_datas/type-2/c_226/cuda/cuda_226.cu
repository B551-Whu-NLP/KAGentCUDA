#include <stdio.h>

// CUDA kernel function
__global__ void histo_atomic(const unsigned int* const vals, unsigned int* const histo, int numVals) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= numVals)
        return;
    atomicAdd(&histo[vals[i]], 1);
}

int main() {
    // Vector size
    int numVals = 100; // Change this according to your requirements

    // Host vectors
    unsigned int* h_vals;
    unsigned int* h_histo;
    h_vals = (unsigned int*)malloc(numVals * sizeof(unsigned int));
    h_histo = (unsigned int*)malloc(numVals * sizeof(unsigned int));

    // Initialize host vectors
    for (int i = 0; i < numVals; ++i) {
        h_vals[i] = i % numVals; // Example data, you can modify this accordingly
    }

    // Device vectors
    unsigned int* d_vals;
    unsigned int* d_histo;
    cudaMalloc((void**)&d_vals, numVals * sizeof(unsigned int));
    cudaMalloc((void**)&d_histo, numVals * sizeof(unsigned int));

    // Copy host vectors to device
    cudaMemcpy(d_vals, h_vals, numVals * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Initialize device histogram to zero
    cudaMemset(d_histo, 0, numVals * sizeof(unsigned int));

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((numVals + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    histo_atomic<<<grid_size, block_size>>>(d_vals, d_histo, numVals);

    // Copy the result back to the host
    cudaMemcpy(h_histo, d_histo, numVals * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Display the result
    for (int i = 0; i < numVals; ++i) {
        printf("%u ", h_histo[i]);
    }
    printf("\n");

    // Clean up
    free(h_vals);
    free(h_histo);
    cudaFree(d_vals);
    cudaFree(d_histo);

    return 0;
}
