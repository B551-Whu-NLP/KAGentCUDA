#include <stdio.h>

// CUDA kernel function
__global__ void doubleArrayVectorElementwiseMultiplyKernel(double* d_in_a, double* d_in_b, double* d_out, int length) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) {
        d_out[tid] = d_in_a[tid] * d_in_b[tid];
    }
}

int main() {
    // Vector size
    int length = 10; // Change this according to your requirements

    // Host arrays
    double* h_in_a = (double*)malloc(length * sizeof(double));
    double* h_in_b = (double*)malloc(length * sizeof(double));
    double* h_out = (double*)malloc(length * sizeof(double));

    // Initialize host input arrays
    for (int i = 0; i < length; ++i) {
        h_in_a[i] = static_cast<double>(i + 1); // Example data, you can modify this accordingly
        h_in_b[i] = static_cast<double>(i);
    }

    // Device arrays
    double* d_in_a;
    double* d_in_b;
    double* d_out;
    cudaMalloc((void**)&d_in_a, length * sizeof(double));
    cudaMalloc((void**)&d_in_b, length * sizeof(double));
    cudaMalloc((void**)&d_out, length * sizeof(double));

    // Copy host input arrays to device
    cudaMemcpy(d_in_a, h_in_a, length * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_b, h_in_b, length * sizeof(double), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((length + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    doubleArrayVectorElementwiseMultiplyKernel<<<grid_size, block_size>>>(d_in_a, d_in_b, d_out, length);

    // Copy the result back to the host
    cudaMemcpy(h_out, d_out, length * sizeof(double), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Result: ");
    for (int i = 0; i < length; ++i) {
        printf("%f ", h_out[i]);
    }
    printf("\n");

    // Clean up
    free(h_in_a);
    free(h_in_b);
    free(h_out);
    cudaFree(d_in_a);
    cudaFree(d_in_b);
    cudaFree(d_out);

    return 0;
}
 
