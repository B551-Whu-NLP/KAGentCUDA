#include <stdio.h>

// CUDA kernel function
__global__ void transpose(float* a, float* b, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        b[col * width + row] = a[row * width + col];
    }
}

int main() {
    // Matrix size
    int width = 4; // Change this according to your requirements

    // Host matrices
    float* h_a = (float*)malloc(width * width * sizeof(float));
    float* h_b = (float*)malloc(width * width * sizeof(float));

    // Initialize host input matrix
    for (int i = 0; i < width * width; ++i) {
        h_a[i] = static_cast<float>(i); // Example data, you can modify this accordingly
    }

    // Device matrices
    float* d_a;
    float* d_b;
    cudaMalloc((void**)&d_a, width * width * sizeof(float));
    cudaMalloc((void**)&d_b, width * width * sizeof(float));

    // Copy host input matrix to device
    cudaMemcpy(d_a, h_a, width * width * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block_size(16, 16); // You can adjust the block size
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (width + block_size.y - 1) / block_size.y);

    // Launch the CUDA kernel
    transpose<<<grid_size, block_size>>>(d_a, d_b, width);

    // Copy the result back to the host
    cudaMemcpy(h_b, d_b, width * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Original Matrix:\n");
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%.2f\t", h_a[i * width + j]);
        }
        printf("\n");
    }

    printf("\nTransposed Matrix:\n");
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%.2f\t", h_b[i * width + j]);
        }
        printf("\n");
    }

    // Clean up
    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
 
