#include <stdio.h>

// CUDA kernel function
__global__ void compareDoubleArrayToThresholdKernel(double* d_in, int* d_out, int length, double threshold) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        double abs_val = (d_in[tid] > 0) ? d_in[tid] : -d_in[tid];
        d_out[tid] = (abs_val < threshold) ? 1 : 0;
    }
}

int main() {
    // Array length
    int length = 100; // Change this according to your requirements

    // Host arrays
    double* h_d_in = (double*)malloc(length * sizeof(double));
    int* h_d_out = (int*)malloc(length * sizeof(int));

    // Initialize host input array
    for (int i = 0; i < length; ++i) {
        h_d_in[i] = (i % 3 == 0) ? 0.5 : -1.0; // Example data, you can modify this accordingly
    }

    // Device arrays
    double* d_d_in;
    int* d_d_out;
    cudaMalloc((void**)&d_d_in, length * sizeof(double));
    cudaMalloc((void**)&d_d_out, length * sizeof(int));

    // Copy host input array to device
    cudaMemcpy(d_d_in, h_d_in, length * sizeof(double), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((length + block_size - 1) / block_size, 1);

    // Threshold value
    double threshold = 1.0; // Change this according to your requirements

    // Launch the CUDA kernel
    compareDoubleArrayToThresholdKernel<<<grid_size, block_size>>>(d_d_in, d_d_out, length, threshold);

    // Copy the result back to the host
    cudaMemcpy(h_d_out, d_d_out, length * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Result: ");
    for (int i = 0; i < length; ++i) {
        printf("%d ", h_d_out[i]);
    }
    printf("\n");

    // Clean up
    free(h_d_in);
    free(h_d_out);
    cudaFree(d_d_in);
    cudaFree(d_d_out);

    return 0;
}
 
