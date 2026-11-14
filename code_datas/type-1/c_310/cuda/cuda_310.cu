#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void smooth_l1_kernel(int n, float *pred, float *truth, float *delta, float *error) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float diff = truth[i] - pred[i];
        float abs_val = fabsf(diff);
        if (abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        } else {
            error[i] = 2 * abs_val - 1;
            delta[i] = (diff > 0) ? 1 : -1;
        }
    }
}

int main() {
    // Set your data size
    const int n = 1024;

    // Allocate host memory
    float *h_pred = (float *)malloc(n * sizeof(float));
    float *h_truth = (float *)malloc(n * sizeof(float));
    float *h_delta = (float *)malloc(n * sizeof(float));
    float *h_error = (float *)malloc(n * sizeof(float));

    // Initialize host data (replace with your data initialization logic)
    for (int i = 0; i < n; ++i) {
        h_pred[i] = static_cast<float>(i);
        h_truth[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float *d_pred, *d_truth, *d_delta, *d_error;
    cudaMalloc((void **)&d_pred, n * sizeof(float));
    cudaMalloc((void **)&d_truth, n * sizeof(float));
    cudaMalloc((void **)&d_delta, n * sizeof(float));
    cudaMalloc((void **)&d_error, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_pred, h_pred, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_truth, h_truth, n * sizeof(float), cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 blockSize(256); // You may adjust the block size
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    // Launch the kernel
    smooth_l1_kernel<<<gridSize, blockSize>>>(n, d_pred, d_truth, d_delta, d_error);

    // Copy result back to host (optional, depends on your application)
    cudaMemcpy(h_delta, d_delta, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_error, d_error, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print or process the result as needed
    printf("Results printed here:\n");

    // Cleanup
    free(h_pred);
    free(h_truth);
    free(h_delta);
    free(h_error);
    cudaFree(d_pred);
    cudaFree(d_truth);
    cudaFree(d_delta);
    cudaFree(d_error);

    return 0;
}
 
