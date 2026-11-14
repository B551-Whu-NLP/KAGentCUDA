#include <iostream>

// CUDA kernel function
__global__ void vectorAdd(double* a, double* b, double* c, int vector_size) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < vector_size) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    // Vector size
    int vector_size = 100; // Change this according to your requirements

    // Host vectors
    double *h_a, *h_b, *h_c;
    h_a = new double[vector_size];
    h_b = new double[vector_size];
    h_c = new double[vector_size];

    // Initialize host vectors
    for (int i = 0; i < vector_size; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Device vectors
    double *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, vector_size * sizeof(double));
    cudaMalloc((void**)&d_b, vector_size * sizeof(double));
    cudaMalloc((void**)&d_c, vector_size * sizeof(double));

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, vector_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, vector_size * sizeof(double), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    int grid_size = (vector_size + block_size - 1) / block_size;

    // Launch the CUDA kernel
    vectorAdd<<<grid_size, block_size>>>(d_a, d_b, d_c, vector_size);

    // Copy the result back to the host
    cudaMemcpy(h_c, d_c, vector_size * sizeof(double), cudaMemcpyDeviceToHost);

    // Display the result
    for (int i = 0; i < vector_size; ++i) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
 
