#include <stdio.h>

// CUDA kernel function
__global__ void rowSumSquareKernel(const double* mat, double* buf, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m) {
        double sum = 0.0;

        for (int j = 0; j < n; j++) {
            double a = mat[i * n + j];
            sum += a * a;
        }

        buf[i] = sum;
    }
}

int main() {
    // Matrix size
    int m = 3;  // Number of rows
    int n = 4;  // Number of columns

    // Host arrays
    double* h_mat = (double*)malloc(m * n * sizeof(double));
    double* h_buf = (double*)malloc(m * sizeof(double));

    // Initialize host input matrix
    for (int i = 0; i < m * n; ++i) {
        h_mat[i] = i;  // Example data for mat, you can modify this accordingly
    }

    // Device arrays
    double* d_mat;
    double* d_buf;
    cudaMalloc((void**)&d_mat, m * n * sizeof(double));
    cudaMalloc((void**)&d_buf, m * sizeof(double));

    // Copy host input matrix to device
    cudaMemcpy(d_mat, h_mat, m * n * sizeof(double), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block_size(2, 1);  // Change this according to your requirements
    dim3 grid_size((m + block_size.x - 1) / block_size.x, 1);

    // Launch the CUDA kernel
    rowSumSquareKernel<<<grid_size, block_size>>>(d_mat, d_buf, m, n);

    // Copy the result back to the host
    cudaMemcpy(h_buf, d_buf, m * sizeof(double), cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Results:\n");
    for (int i = 0; i < m; ++i) {
        printf("%f ", h_buf[i]);
    }
    printf("\n");

    // Clean up
    free(h_mat);
    free(h_buf);
    cudaFree(d_mat);
    cudaFree(d_buf);

    return 0;
}
 
