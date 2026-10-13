#include <stdio.h>

// CUDA kernel function
__global__ void update_x(double* x, double* a, double* b, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride) {
        x[i] = 2.0 / 3.0 * a[i] / b[i] + 1.0 / 3.0 * x[i];
    }
}

int main() {
    // Array size
    int n = 100; // Change this according to your requirements

    // Host arrays
    double* h_x = (double*)malloc(n * sizeof(double));
    double* h_a = (double*)malloc(n * sizeof(double));
    double* h_b = (double*)malloc(n * sizeof(double));

    // Initialize host input arrays
    for (int i = 0; i < n; ++i) {
        h_x[i] = i + 1.0; // Example data for x, you can modify this accordingly
        h_a[i] = 2.0 * i + 1.0; // Example data for a, you can modify this accordingly
        h_b[i] = i + 2.0; // Example data for b, you can modify this accordingly
    }

    // Device arrays
    double* d_x;
    double* d_a;
    double* d_b;
    cudaMalloc((void**)&d_x, n * sizeof(double));
    cudaMalloc((void**)&d_a, n * sizeof(double));
    cudaMalloc((void**)&d_b, n * sizeof(double));

    // Copy host input arrays to device
    cudaMemcpy(d_x, h_x, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(double), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((n + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    update_x<<<grid_size, block_size>>>(d_x, d_a, d_b, n);

    // Copy the result back to the host
    cudaMemcpy(h_x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Initial x Array:\n");
    for (int i = 0; i < n; ++i) {
        printf("%.2f ", h_x[i]);
    }

    printf("\n");

    // Clean up
    free(h_x);
    free(h_a);
    free(h_b);
    cudaFree(d_x);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
 
