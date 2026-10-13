#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matrixProduct(double *matrix_a, double *matrix_b, double *matrix_c, int width, int from, int my_rank) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    matrix_c[row * width + col] = 0;
    for (int k = 0; k < width; k++) {
        matrix_c[row * width + col] += matrix_a[((row + from) * width) + k] * matrix_b[k * width + col];
    }
}

int main() {
    // Set your problem dimensions
    const int width = 256;  // Adjust as needed
    const int height = 256; // Adjust as needed
    const int from = 0;     // Adjust as needed
    const int my_rank = 0;  // Adjust as needed

    // Allocate host memory
    double *h_matrix_a = (double *)malloc(width * height * sizeof(double));
    double *h_matrix_b = (double *)malloc(width * width * sizeof(double));
    double *h_matrix_c = (double *)malloc(width * height * sizeof(double));

    // Initialize host data (replace with your data initialization logic)
    for (int i = 0; i < width * height; ++i) {
        h_matrix_a[i] = static_cast<double>(i);
    }

    for (int i = 0; i < width * width; ++i) {
        h_matrix_b[i] = static_cast<double>(i);
    }

    // Allocate device memory
    double *d_matrix_a, *d_matrix_b, *d_matrix_c;
    cudaMalloc((void **)&d_matrix_a, width * height * sizeof(double));
    cudaMalloc((void **)&d_matrix_b, width * width * sizeof(double));
    cudaMalloc((void **)&d_matrix_c, width * height * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_matrix_a, h_matrix_a, width * height * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, h_matrix_b, width * width * sizeof(double), cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 blockSize(16, 16); // You may adjust the block size
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    matrixProduct<<<gridSize, blockSize>>>(d_matrix_a, d_matrix_b, d_matrix_c, width, from, my_rank);

    // Copy result back to host (optional, depends on your application)
    cudaMemcpy(h_matrix_c, d_matrix_c, width * height * sizeof(double), cudaMemcpyDeviceToHost);

    // Print or process the result as needed
    printf("Results printed here:\n");

    // Cleanup
    free(h_matrix_a);
    free(h_matrix_b);
    free(h_matrix_c);
    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_matrix_c);

    return 0;
}
 
