#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void Kernel_Transpose2d(float* dev_transposeArray, float* dev_array, const int r, const int c) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= r || j >= c)
        return;

    int idx_transposeArray, idx_array;
    idx_array = i * c + j;
    idx_transposeArray = j * r + i;
    dev_transposeArray[idx_transposeArray] = dev_array[idx_array];
}

int main() {
    // Set the dimensions of the matrix
    const int r = 4; // Rows
    const int c = 3; // Columns

    // Host arrays
    float* h_dev_transposeArray = (float*)malloc(r * c * sizeof(float));
    float* h_dev_array = (float*)malloc(r * c * sizeof(float));

    // Initialize host arrays (example data, modify as needed)
    for (int i = 0; i < r * c; ++i) {
        h_dev_array[i] = i;
    }

    // Device arrays
    float* d_dev_transposeArray;
    float* d_dev_array;

    cudaMalloc((void**)&d_dev_transposeArray, r * c * sizeof(float));
    cudaMalloc((void**)&d_dev_array, r * c * sizeof(float));

    // Copy host data to device
    cudaMemcpy(d_dev_array, h_dev_array, r * c * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block_size(16, 16); // Adjust this according to your requirements
    dim3 grid_size((r + block_size.x - 1) / block_size.x, (c + block_size.y - 1) / block_size.y);

    // Launch the CUDA kernel
    Kernel_Transpose2d<<<grid_size, block_size>>>(d_dev_transposeArray, d_dev_array, r, c);

    // Copy the result back to the host (optional)
    cudaMemcpy(h_dev_transposeArray, d_dev_transposeArray, r * c * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Original Array:\n");
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            printf("%.2f\t", h_dev_array[i * c + j]);
        }
        printf("\n");
    }

    printf("\nTransposed Array:\n");
    for (int i = 0; i < c; ++i) {
        for (int j = 0; j < r; ++j) {
            printf("%.2f\t", h_dev_transposeArray[i * r + j]);
        }
        printf("\n");
    }

    // Clean up
    free(h_dev_transposeArray);
    free(h_dev_array);
    cudaFree(d_dev_transposeArray);
    cudaFree(d_dev_array);

    return 0;
}
 
