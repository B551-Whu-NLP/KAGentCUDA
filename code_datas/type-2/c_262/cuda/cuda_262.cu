#include <stdio.h>

// CUDA kernel function
__global__ void kernel_sum_backward(float* db, float* dout, int r, int c) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int N = c;

    while (tid < N) {
        for (int i = 0; i < r; i++) {
            db[tid] += dout[i * c + tid];
        }
        tid += gridDim.x * blockDim.x;
    }
}

int main() {
    // Matrix size
    int r = 4; // Number of rows
    int c = 5; // Number of columns

    // Host arrays
    float* h_db = (float*)malloc(c * sizeof(float));
    float* h_dout = (float*)malloc(r * c * sizeof(float));

    // Initialize host input arrays
    for (int i = 0; i < c; ++i) {
        h_db[i] = 0.0; // Initialize db with zeros
    }

    for (int i = 0; i < r * c; ++i) {
        h_dout[i] = i + 1; // Example data, you can modify this accordingly
    }

    // Device arrays
    float* d_db;
    float* d_dout;
    cudaMalloc((void**)&d_db, c * sizeof(float));
    cudaMalloc((void**)&d_dout, r * c * sizeof(float));

    // Copy host input arrays to device
    cudaMemcpy(d_db, h_db, c * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dout, h_dout, r * c * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((c + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    kernel_sum_backward<<<grid_size, block_size>>>(d_db, d_dout, r, c);

    // Copy the result back to the host
    cudaMemcpy(h_db, d_db, c * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Gradients with respect to input (db):\n");
    for (int i = 0; i < c; ++i) {
        printf("%.2f ", h_db[i]);
    }
    printf("\n");

    // Clean up
    free(h_db);
    free(h_dout);
    cudaFree(d_db);
    cudaFree(d_dout);

    return 0;
}
 
