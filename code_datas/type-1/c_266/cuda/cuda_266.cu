#include <stdio.h>
#include <math.h>

// CUDA kernel function
__global__ void l2_kernel(int n, float* pred, float* truth, float* delta, float* error) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < n) {
        float diff = truth[i] - pred[i];
        error[i] = diff * diff;
        delta[i] = diff;
    }
}

int main() {
    // Array size
    int n = 100; // Change this according to your requirements

    // Host arrays
    float* h_pred = (float*)malloc(n * sizeof(float));
    float* h_truth = (float*)malloc(n * sizeof(float));
    float* h_delta = (float*)malloc(n * sizeof(float));
    float* h_error = (float*)malloc(n * sizeof(float));

    // Initialize host input arrays
    for (int i = 0; i < n; ++i) {
        h_pred[i] = i; // Example data for pred, you can modify this accordingly
        h_truth[i] = 2 * i; // Example data for truth, you can modify this accordingly
    }

    // Device arrays
    float* d_pred;
    float* d_truth;
    float* d_delta;
    float* d_error;
    cudaMalloc((void**)&d_pred, n * sizeof(float));
    cudaMalloc((void**)&d_truth, n * sizeof(float));
    cudaMalloc((void**)&d_delta, n * sizeof(float));
    cudaMalloc((void**)&d_error, n * sizeof(float));

    // Copy host input arrays to device
    cudaMemcpy(d_pred, h_pred, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_truth, h_truth, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((n + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    l2_kernel<<<grid_size, block_size>>>(n, d_pred, d_truth, d_delta, d_error);

    // Copy the result back to the host
    cudaMemcpy(h_delta, d_delta, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_error, d_error, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Pred Array:\n");
    for (int i = 0; i < n; ++i) {
        printf("%.2f ", h_pred[i]);
    }

    printf("\nTruth Array:\n");
    for (int i = 0; i < n; ++i) {
        printf("%.2f ", h_truth[i]);
    }

    printf("\nDelta Array:\n");
    for (int i = 0; i < n; ++i) {
        printf("%.2f ", h_delta[i]);
    }

    printf("\nError Array:\n");
    for (int i = 0; i < n; ++i) {
        printf("%.2f ", h_error[i]);
    }
    printf("\n");

    // Clean up
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
 
