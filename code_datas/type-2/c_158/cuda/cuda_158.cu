#include <stdio.h>

// Define the CUDA kernel
__global__ void ConvLayerForward_Kernel(int C, int W_grid, int K, float *X, float *W, float *Y) {
    int n, m, h, w, c, p, q;
    
    n = blockIdx.x;
    m = blockIdx.y;
    h = blockIdx.z / W_grid + threadIdx.y;
    w = blockIdx.z % W_grid + threadIdx.x;

    float acc = 0;

    for (c = 0; c < C; c++) {
        for (p = 0; p < K; p++) {
            for (q = 0; q < K; q++) {
                acc += X[n * C * W_grid * W_grid + c * W_grid * W_grid + (h + p) * W_grid + (w + q)] * W[m * C * K * K + c * K * K + p * K + q];
            }
        }
    }

    Y[n * W_grid * W_grid * W_grid + m * W_grid * W_grid + h * W_grid + w] = acc;
}

int main() {
    // Example usage
    int C = 3, W_grid = 4, K = 3;  // Set your values accordingly
    float *X, *W, *Y;  // Assuming these arrays are allocated and initialized

    // Set the CUDA device
    cudaSetDevice(0);

    // Allocate device memory
    float *d_X, *d_W, *d_Y;
    cudaMalloc((void **)&d_X, C * W_grid * W_grid * sizeof(float));
    cudaMalloc((void **)&d_W, C * K * K * sizeof(float));
    cudaMalloc((void **)&d_Y, W_grid * W_grid * W_grid * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_X, X, C * W_grid * W_grid * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, C * K * K * sizeof(float), cudaMemcpyHostToDevice);

    // Configure the CUDA kernel launch
    dim3 threadsPerBlock(K, K);
    dim3 blocksPerGrid(W_grid, W_grid, W_grid);

    // Launch the CUDA kernel
    ConvLayerForward_Kernel<<<blocksPerGrid, threadsPerBlock>>>(C, W_grid, K, d_X, d_W, d_Y);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(Y, d_Y, W_grid * W_grid * W_grid * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_Y);

    return 0;
}
