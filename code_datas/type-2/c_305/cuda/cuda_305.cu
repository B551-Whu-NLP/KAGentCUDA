#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void manage_adj_matrix(float *gpu_graph, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float sum = 0.0;

        for (int i = 0; i < n; ++i) {
            sum += gpu_graph[i * n + id];
        }

        for (int i = 0; i < n; ++i) {
            if (sum != 0.0) {
                gpu_graph[i * n + id] /= sum;
            } else {
                gpu_graph[i * n + id] = (1.0 / (float)n);
            }
        }
    }
}

int main() {
    // Set your problem dimensions
    const int n = 256;

    // Allocate host memory
    float *h_gpu_graph = (float *)malloc(n * n * sizeof(float));

    // Initialize host data (replace with your data initialization logic)
    for (int i = 0; i < n * n; ++i) {
        h_gpu_graph[i] = static_cast<float>(i % 10);
    }

    // Allocate device memory
    float *d_gpu_graph;
    cudaMalloc((void **)&d_gpu_graph, n * n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_gpu_graph, h_gpu_graph, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 blockSize(256);  // You may adjust the block size
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    // Launch the kernel
    manage_adj_matrix<<<gridSize, blockSize>>>(d_gpu_graph, n);

    // Copy result back to host (optional, depends on your application)
    cudaMemcpy(h_gpu_graph, d_gpu_graph, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print or process the result as needed
    printf("Results printed here:\n");

    // Cleanup
    free(h_gpu_graph);
    cudaFree(d_gpu_graph);

    return 0;
}
 
