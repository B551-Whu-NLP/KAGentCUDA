#include <stdio.h>

// CUDA kernel function
__global__ void naiveParTrans(float* d_in, float* d_out, int x_size, int y_size) {
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gidy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gidx < x_size && gidy < y_size) {
        d_out[gidx * y_size + gidy] = d_in[gidy * x_size + gidx];
    }
}

int main() {
    // Matrix dimensions
    int x_size = 4; // Number of rows
    int y_size = 3; // Number of columns

    // Host matrices
    float* h_in = (float*)malloc(x_size * y_size * sizeof(float));
    float* h_out = (float*)malloc(y_size * x_size * sizeof(float));

    // Initialize host input matrix
    for (int i = 0; i < x_size * y_size; ++i) {
        h_in[i] = i + 1.0; // Example data, you can modify this accordingly
    }

    // Device matrices
    float* d_in;
    float* d_out;
    cudaMalloc((void**)&d_in, x_size * y_size * sizeof(float));
    cudaMalloc((void**)&d_out, y_size * x_size * sizeof(float));

    // Copy host input matrix to device
    cudaMemcpy(d_in, h_in, x_size * y_size * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block_size(2, 2); // Change this according to your requirements
    dim3 grid_size((x_size + block_size.x - 1) / block_size.x, (y_size + block_size.y - 1) / block_size.y);

    // Launch the CUDA kernel
    naiveParTrans<<<grid_size, block_size>>>(d_in, d_out, x_size, y_size);

    // Copy the result back to the host
    cudaMemcpy(h_out, d_out, y_size * x_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Input Matrix:\n");
    for (int i = 0; i < x_size * y_size; ++i) {
        printf("%.2f ", h_in[i]);
        if ((i + 1) % y_size == 0) {
            printf("\n");
        }
    }

    printf("\nTransposed Matrix:\n");
    for (int i = 0; i < y_size * x_size; ++i) {
        printf("%.2f ", h_out[i]);
        if ((i + 1) % x_size == 0) {
            printf("\n");
        }
    }

    // Clean up
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
 
