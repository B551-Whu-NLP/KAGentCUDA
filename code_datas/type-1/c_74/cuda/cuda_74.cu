#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void copy_array_d2d(double** src, double** dst, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n) {
        dst[i][j] = src[i][j];
    }
}

int main() {
    const int m = 10; // Rows
    const int n = 5;  // Columns

    // Allocate and initialize 2D arrays on the host
    double** h_src = new double*[m];
    double** h_dst = new double*[m];
    for (int i = 0; i < m; ++i) {
        h_src[i] = new double[n];
        h_dst[i] = new double[n];
        for (int j = 0; j < n; ++j) {
            h_src[i][j] = i * n + j; // Some example data
        }
    }

    // Allocate device memory
    double** d_src;
    double** d_dst;
    cudaMalloc((void**)&d_src, m * sizeof(double*));
    cudaMalloc((void**)&d_dst, m * sizeof(double*));

    for (int i = 0; i < m; ++i) {
        cudaMalloc((void**)&d_src[i], n * sizeof(double));
        cudaMalloc((void**)&d_dst[i], n * sizeof(double));
        cudaMemcpy(d_src[i], h_src[i], n * sizeof(double), cudaMemcpyHostToDevice);
    }

    // Launch the kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((m + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);
    copy_array_d2d<<<gridSize, blockSize>>>(d_src, d_dst, m, n);

    // Copy the result back to the host
    for (int i = 0; i < m; ++i) {
        cudaMemcpy(h_dst[i], d_dst[i], n * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // Print the result
    printf("Original array:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", h_src[i][j]);
        }
        printf("\n");
    }

    printf("\nCopied array:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", h_dst[i][j]);
        }
        printf("\n");
    }

    // Cleanup
    for (int i = 0; i < m; ++i) {
        cudaFree(d_src[i]);
        cudaFree(d_dst[i]);
        delete[] h_src[i];
        delete[] h_dst[i];
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    delete[] h_src;
    delete[] h_dst;

    return 0;
}
