#include <stdio.h>

// CUDA kernel function
__global__ void fill_idx(int N, int* device_input, int* device_output) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx + 1 < N && device_input[idx] + 1 == device_input[idx + 1]) {
        device_output[device_input[idx]] = idx;
    }
}

int main() {
    // Vector size
    int N = 10; // Change this according to your requirements

    // Host arrays
    int* h_input = (int*)malloc(N * sizeof(int));
    int* h_output = (int*)malloc(N * sizeof(int));

    // Initialize host input array
    for (int i = 0; i < N; ++i) {
        h_input[i] = i; // Example data, you can modify this accordingly
    }

    // Device arrays
    int* d_input;
    int* d_output;
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, N * sizeof(int));

    // Copy host input array to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((N + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    fill_idx<<<grid_size, block_size>>>(N, d_input, d_output);

    // Copy the result back to the host
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Result: ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    // Clean up
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
 
