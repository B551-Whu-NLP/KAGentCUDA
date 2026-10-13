#include <stdio.h>

// CUDA kernel function
__global__ void reverseArrayBlock(int* d_out, int* d_in) {
    extern __shared__ int s_data[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = (gridDim.x - 1 - blockIdx.x) * blockDim.x + threadIdx.x;

    s_data[blockDim.x - 1 - threadIdx.x] = d_in[i];

    __syncthreads();

    d_out[j] = s_data[threadIdx.x];
}

int main() {
    // Array size
    int size = 10; // Change this according to your requirements

    // Host arrays
    int* h_in = (int*)malloc(size * sizeof(int));
    int* h_out = (int*)malloc(size * sizeof(int));

    // Initialize host input array
    for (int i = 0; i < size; ++i) {
        h_in[i] = i; // Example data, you can modify this accordingly
    }

    // Device arrays
    int* d_in;
    int* d_out;
    cudaMalloc((void**)&d_in, size * sizeof(int));
    cudaMalloc((void**)&d_out, size * sizeof(int));

    // Copy host input array to device
    cudaMemcpy(d_in, h_in, size * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((size + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel with shared memory
    reverseArrayBlock<<<grid_size, block_size, block_size * sizeof(int)>>>(d_out, d_in);

    // Copy the result back to the host
    cudaMemcpy(h_out, d_out, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Original Array: ");
    for (int i = 0; i < size; ++i) {
        printf("%d ", h_in[i]);
    }
    printf("\n");

    printf("Reversed Array: ");
    for (int i = 0; i < size; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    // Clean up
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
 
