#include <stdio.h>
#include <math.h>

// CUDA kernel function
__global__ void kernel(float* x, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        double sum = 0;

        for (int j = 0; j < 1000; j++) {
            sum += sqrt(pow(3.14159, i)) / float(j);
        }

        x[i] = sum;
    }
}

int main() {
    // Array size
    int size = 1024;  // Change this according to your requirements

    // Host array
    float* h_x = (float*)malloc(size * sizeof(float));

    // Device array
    float* d_x;
    cudaMalloc((void**)&d_x, size * sizeof(float));

    // Define block and grid dimensions
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    // Launch the CUDA kernel
    kernel<<<grid_size, block_size>>>(d_x, size);

    // Copy the result back to the host
    cudaMemcpy(h_x, d_x, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Results:\n");
    for (int i = 0; i < size; ++i) {
        printf("%f ", h_x[i]);
    }
    printf("\n");

    // Clean up
    free(h_x);
    cudaFree(d_x);

    return 0;
}
 
