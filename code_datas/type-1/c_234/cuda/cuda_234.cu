#include <stdio.h>

// CUDA kernel function
__global__ void doubleArraySignKernel(double* d_in, double* d_out, int length) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < length) {
        d_out[tid] = (0 < d_in[tid]) - (d_in[tid] < 0);
    }
}

int main() {
    // Vector size
    int length = 100; // Change this according to your requirements

    // Host vectors
    double* h_d_in;
    double* h_d_out;
    h_d_in = (double*)malloc(length * sizeof(double));
    h_d_out = (double*)malloc(length * sizeof(double));

    // Initialize host vectors
    for (int i = 0; i < length; ++i) {
        h_d_in[i] = i - 50.0; // Example data, you can modify this accordingly
    }

    // Device vectors
    double* d_d_in;
    double* d_d_out;
    cudaMalloc((void**)&d_d_in, length * sizeof(double));
    cudaMalloc((void**)&d_d_out, length * sizeof(double));

    // Copy host vectors to device
    cudaMemcpy(d_d_in, h_d_in, length * sizeof(double), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((length + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    doubleArraySignKernel<<<grid_size, block_size>>>(d_d_in, d_d_out, length);

    // Copy the result back to the host
    cudaMemcpy(h_d_out, d_d_out, length * sizeof(double), cudaMemcpyDeviceToHost);

    // Display the result
    for (int i = 0; i < length; ++i) {
        printf("%f ", h_d_out[i]);
    }
    printf("\n");

    // Clean up
    free(h_d_in);
    free(h_d_out);
    cudaFree(d_d_in);
    cudaFree(d_d_out);

    return 0;
}
 
