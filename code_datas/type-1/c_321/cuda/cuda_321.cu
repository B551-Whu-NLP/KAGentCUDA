#include <stdio.h>
#include <cuda_runtime.h>

__global__ void create_p_vect(float *node_info1, float *node_info2, float *p, int n_nodes_1, int n_nodes_2) {
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    int ty = threadIdx.y + blockDim.y * blockIdx.y;
    float cutoff = 0.5;

    if ((tx < n_nodes_1) && (ty < n_nodes_2)) {
        int ind = tx * n_nodes_2 + ty;

        if ((node_info1[tx] < cutoff) && (node_info2[ty] < cutoff))
            p[ind] = 0;
        else
            p[ind] = node_info1[tx] * node_info2[ty];
    }
}

int main() {
    // Set matrix dimensions
    int n_nodes_1 = 4;  // Set the appropriate value
    int n_nodes_2 = 3;  // Set the appropriate value

    // Allocate host memory
    float *h_node_info1, *h_node_info2, *h_p;
    h_node_info1 = (float *)malloc(n_nodes_1 * sizeof(float));
    h_node_info2 = (float *)malloc(n_nodes_2 * sizeof(float));
    h_p = (float *)malloc(n_nodes_1 * n_nodes_2 * sizeof(float));

    // Initialize arrays (you may use your own initialization logic)
    for (int i = 0; i < n_nodes_1; i++) {
        h_node_info1[i] = static_cast<float>(i + 1) / 10.0f;
    }

    for (int i = 0; i < n_nodes_2; i++) {
        h_node_info2[i] = static_cast<float>(i + 1) / 10.0f;
    }

    // Allocate device memory
    float *d_node_info1, *d_node_info2, *d_p;
    cudaMalloc((void **)&d_node_info1, n_nodes_1 * sizeof(float));
    cudaMalloc((void **)&d_node_info2, n_nodes_2 * sizeof(float));
    cudaMalloc((void **)&d_p, n_nodes_1 * n_nodes_2 * sizeof(float));

    // Copy arrays from host to device
    cudaMemcpy(d_node_info1, h_node_info1, n_nodes_1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_info2, h_node_info2, n_nodes_2 * sizeof(float), cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 blockSize(16, 16);  // You may adjust the block size
    dim3 gridSize((n_nodes_1 + blockSize.x - 1) / blockSize.x, (n_nodes_2 + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    create_p_vect<<<gridSize, blockSize>>>(d_node_info1, d_node_info2, d_p, n_nodes_1, n_nodes_2);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Check for errors during the kernel launch
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    // Copy the result array from device to host
    cudaMemcpy(h_p, d_p, n_nodes_1 * n_nodes_2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result array (you may modify this part based on your needs)
    printf("Result Array:\n");
    for (int i = 0; i < n_nodes_1; i++) {
        for (int j = 0; j < n_nodes_2; j++) {
            printf("%.2f ", h_p[i * n_nodes_2 + j]);
        }
        printf("\n");
    }

    // Cleanup
    free(h_node_info1);
    free(h_node_info2);
    free(h_p);
    cudaFree(d_node_info1);
    cudaFree(d_node_info2);
    cudaFree(d_p);

    return 0;
}
 
