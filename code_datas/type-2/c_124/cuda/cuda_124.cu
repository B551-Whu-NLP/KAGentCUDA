#include <stdio.h>

// CUDA kernel
__global__ void gather_points_kernel(int b, int c, int n, int m, const float* __restrict__ points, const int* __restrict__ idx, float* __restrict__ out) {
    for (int i = blockIdx.x; i < b; i += gridDim.x) {
        for (int l = blockIdx.y; l < c; l += gridDim.y) {
            for (int j = threadIdx.x; j < m; j += blockDim.x) {
                int a = idx[i * m + j];
                out[(i * c + l) * m + j] = points[(i * c + l) * n + a];
            }
        }
    }
}

int main() {
    // Your main program logic here

    // Example: Allocate and initialize host and device memory
    int b = 100; // Replace with your actual values
    int c = 3;
    int n = 500;
    int m = 10;

    float* h_points = (float*)malloc(b * c * n * sizeof(float));
    int* h_idx = (int*)malloc(b * m * sizeof(int));
    float* h_out = (float*)malloc(b * c * m * sizeof(float));

    float* d_points, * d_out;
    int* d_idx;

    cudaMalloc((void**)&d_points, b * c * n * sizeof(float));
    cudaMalloc((void**)&d_idx, b * m * sizeof(int));
    cudaMalloc((void**)&d_out, b * c * m * sizeof(float));

    // Copy host memory to device
    cudaMemcpy(d_points, h_points, b * c * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, h_idx, b * m * sizeof(int), cudaMemcpyHostToDevice);

    // Configure and launch the CUDA kernel
    dim3 blockSize(256); // Adjust block dimensions based on your requirements
    dim3 gridSize(b, c);  // Adjust grid dimensions based on your requirements
    gather_points_kernel<<<gridSize, blockSize>>>(b, c, n, m, d_points, d_idx, d_out);

    // Copy device memory back to host
    cudaMemcpy(h_out, d_out, b * c * m * sizeof(float), cudaMemcpyDeviceToHost);

    // Your post-kernel logic here

    // Free allocated memory
    free(h_points);
    free(h_idx);
    free(h_out);
    cudaFree(d_points);
    cudaFree(d_idx);
    cudaFree(d_out);

    return 0;
}
