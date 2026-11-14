#include <stdio.h>
#include <cmath>

// CUDA runtime
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void logistic_x_ent_kernel(int n, float* pred, float* truth, float* delta, float* error) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i < n) {
        float t = truth[i];
        float p = pred[i];
        error[i] = -t * log(p + 0.0000001f) - (1 - t) * log(1 - p + 0.0000001f);
        delta[i] = t - p;
    }
}

int main() {
    // Set the parameters
    const int n = 1024; // Change as needed

    // Host arrays
    float* h_pred = (float*)malloc(n * sizeof(float));
    float* h_truth = (float*)malloc(n * sizeof(float));
    float* h_delta = (float*)malloc(n * sizeof(float));
    float* h_error = (float*)malloc(n * sizeof(float));

    // Initialize host arrays (example data, modify as needed)
    for (int i = 0; i < n; ++i) {
        h_pred[i] = 0.7f;  // Example data, you can modify this accordingly
        h_truth[i] = 0.5f; // Example data, you can modify this accordingly
    }

    // Device arrays
    float *d_pred, *d_truth, *d_delta, *d_error;

    cudaMalloc((void**)&d_pred, n * sizeof(float));
    cudaMalloc((void**)&d_truth, n * sizeof(float));
    cudaMalloc((void**)&d_delta, n * sizeof(float));
    cudaMalloc((void**)&d_error, n * sizeof(float));

    // Copy host data to device
    cudaMemcpy(d_pred, h_pred, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_truth, h_truth, n * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Launch the CUDA kernel
    logistic_x_ent_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_pred, d_truth, d_delta, d_error);

    // Copy the result back to the host (optional)
    cudaMemcpy(h_delta, d_delta, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_error, d_error, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Delta and Error:\n");
    for (int i = 0; i < n; ++i) {
        printf("Element %d: Delta=%f, Error=%f\n", i, h_delta[i], h_error[i]);
    }

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
 
