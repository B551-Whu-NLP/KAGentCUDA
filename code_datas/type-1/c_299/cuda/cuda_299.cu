#include <stdio.h>

__global__ void check_results_kernel(uint* g_results0, uint* g_results1, int n) {
    uint idx = threadIdx.x;
    uint gidx = blockDim.x * blockIdx.x + idx;
    uint result0, result1;

    if (gidx < n) {
        result0 = g_results0[gidx];
        result1 = g_results1[gidx];

        if (result0 != result1) {
            printf("%d != %d for thread %d\n", result0, result1, gidx);
        }
    }
}

int main() {
    // Set the parameters
    const int n = 1000;  // Adjust the size as needed

    // Host arrays
    uint* h_results0 = (uint*)malloc(n * sizeof(uint));
    uint* h_results1 = (uint*)malloc(n * sizeof(uint));

    // Initialize host arrays (example data, modify as needed)
    for (int i = 0; i < n; ++i) {
        h_results0[i] = i;
        h_results1[i] = i * 2;  // Mismatch for illustration
    }

    // Device arrays
    uint* d_results0;
    uint* d_results1;

    cudaMalloc((void**)&d_results0, n * sizeof(uint));
    cudaMalloc((void**)&d_results1, n * sizeof(uint));

    // Copy host data to device
    cudaMemcpy(d_results0, h_results0, n * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_results1, h_results1, n * sizeof(uint), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x
 
