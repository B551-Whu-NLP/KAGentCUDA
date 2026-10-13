#include <stdio.h>

// CUDA kernel function
__global__ void matVecRowSubKernel(const double* mat, const double* vec, double* buf, int m, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < m * n) {
        int i = index / n;
        int j = index % n;
        buf[i * n + j] = mat[i * n + j] - vec[j];
    }
}

int main() {
    // Matrix size
    int m = 4;  // Change this according to your requirements

    // Vector size
    int n = 3;  // Change this according to your requirements

    // Host arrays
    double* h_mat = (double*)malloc(m * n * sizeof(double));
    double* h_vec = (double*)malloc(n * sizeof(double));
    double* h_buf = (double*)malloc(m * n * sizeof(double));

    // Initialize host input arrays
    for (int i = 0; i < m * n; ++i) {
        h_mat[i] = i;  // Example data for mat, you can modify this accordingly
    }

    for (int i = 0; i < n; ++i) {
        h_vec[i] = i;  // Example data for vec, you can modify this accordingly
    }

    // Device arrays
    double* d_mat;
    double* d_vec;
    double* d_buf;
    cudaMalloc((void**)&d_mat, m * n * sizeof(double));
    cudaMalloc((void**)&d_vec, n * sizeof(double));
    cudaMalloc((void**)&d_buf, m * n * sizeof(double));

    // Copy host input arrays to device
    cudaMemcpy(d_mat, h_mat, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec, n * sizeof(double), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    int grid_size = (m * n + block_size - 1) / block_size;

    // Launch the CUDA kernel
    matVecRowSubKernel<<<grid_size, block_size>>>(d_mat, d_vec, d_buf, m, n);

    // Copy the result back to the host
    cudaMemcpy(h_buf, d_buf, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Results:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", h_buf[i * n + j]);
        }
        printf("\n");
    }

    // Clean up
    free(h_mat);
    free(h_vec);
    free(h_buf);
    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(d_buf);

    return 0;
}
 
