#include <stdio.h>
#include <cuda_runtime.h>

__global__ void calcbidvalues(int n, int *src2tgt, float *adj, float *prices, bool *complete, float *values, float *bids) {
    int INDEX = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = INDEX; idx < n * n; idx += stride) {
        int i = idx / n;
        int j = idx - i * n;
        bids[i * n + j] = -1;

        if (src2tgt[i] != -1) {
            continue;
        }

        complete[0] = false;
        values[i * n + j] = -adj[i * n + j] - prices[j];
    }
}

int main() {
    // Set matrix dimensions
    int n = 4; // Set the appropriate value

    // Allocate host memory
    int *h_src2tgt;
    float *h_adj, *h_prices, *h_values, *h_bids;
    bool *h_complete;

    h_src2tgt = (int *)malloc(n * sizeof(int));
    h_adj = (float *)malloc(n * n * sizeof(float));
    h_prices = (float *)malloc(n * sizeof(float));
    h_values = (float *)malloc(n * n * sizeof(float));
    h_bids = (float *)malloc(n * n * sizeof(float));
    h_complete = (bool *)malloc(sizeof(bool));

    // Initialize arrays (you may use your own initialization logic)
    for (int i = 0; i < n; i++) {
        h_src2tgt[i] = -1; // Initialize with appropriate values
        h_prices[i] = 0.5;  // Initialize with appropriate values

        for (int j = 0; j < n; j++) {
            h_adj[i * n + j] = static_cast<float>(i + j) / 10.0f; // Initialize with appropriate values
        }
    }

    // Allocate device memory
    int *d_src2tgt;
    float *d_adj, *d_prices, *d_values, *d_bids;
    bool *d_complete;

    cudaMalloc((void **)&d_src2tgt, n * sizeof(int));
    cudaMalloc((void **)&d_adj, n * n * sizeof(float));
    cudaMalloc((void **)&d_prices, n * sizeof(float));
    cudaMalloc((void **)&d_values, n * n * sizeof(float));
    cudaMalloc((void **)&d_bids, n * n * sizeof(float));
    cudaMalloc((void **)&d_complete, sizeof(bool));

    // Copy arrays from host to device
    cudaMemcpy(d_src2tgt, h_src2tgt, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adj, h_adj, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prices, h_prices, n * sizeof(float), cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 blockSize(16, 16); // You may adjust the block size
    dim3 gridSize((n * n + blockSize.x - 1) / blockSize.x, 1);

    // Launch the kernel
    calcbidvalues<<<gridSize, blockSize>>>(n, d_src2tgt, d_adj, d_prices, d_complete, d_values, d_bids);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Check for errors during the kernel launch
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    // Copy the result arrays from device to host
    cudaMemcpy(h_values, d_values, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bids, d_bids, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_complete, d_complete, sizeof(bool), cudaMemcpyDeviceToHost);

    // Display the result arrays (you may modify this part based on your needs)
    printf("Values Array:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", h_values[i * n + j]);
        }
        printf("\n");
    }

    printf("Bids Array:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", h_bids[i * n + j]);
        }
        printf("\n");
    }

    printf("Complete Flag: %s\n", *h_complete ? "true" : "false");

    // Cleanup
    free(h_src2tgt);
    free(h_adj);
    free(h_prices);
    free(h_values);
    free(h_bids);
    free(h_complete
 
