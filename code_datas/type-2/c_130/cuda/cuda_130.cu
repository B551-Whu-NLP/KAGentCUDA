#include <stdio.h>

// CUDA kernel
__global__ void colLog2SumExp2Kernel(const double* mat, double* buf, int m, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < n) {
        double maximum = mat[j];

        for (int i = 1; i < m; i++) {
            if (mat[i * n + j] > maximum) {
                maximum = mat[i * n + j];
            }
        }

        double res = 0.0;

        for (int i = 0; i < m; i++) {
            res += mat[i * n + j] - maximum;
        }

        buf[j] = res + maximum;
    }
}

int main() {
    // Your main program logic here

    // Example: Allocate and initialize host and device memory
    int m = 100; // Replace with your actual size
    int n = 50;  // Replace with your actual size

    double* h_mat = (double*)malloc(m * n * sizeof(double));
    double* h_buf = (double*)malloc(n * sizeof(double));

    double* d_mat, * d_buf;
    cudaMalloc((void**)&d_mat, m * n * sizeof(double));
    cudaMalloc((void**)&d_buf, n * sizeof(double));

    // Copy host memory to device
    cudaMemcpy(d_mat, h_mat, m * n * sizeof(double), cudaMemcpyHostToDevice);

    // Configure and launch the CUDA kernel
    dim3 blockSize(256); // Adjust block dimensions based on your requirements
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    colLog2SumExp2Kernel<<<gridSize, blockSize>>>(d_mat, d_buf, m, n);

    // Copy device memory back to host
    cudaMemcpy(h_buf, d_buf, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Your post-kernel logic here

    // Free allocated memory
    free(h_mat);
    free(h_buf);
    cudaFree(d_mat);
    cudaFree(d_buf);

    return 0;
}
