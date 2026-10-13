#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void Kernel_Argmax(int* dev_argMax, float* dev_array, const int r, const int c) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= r) return;

    int idx;
    float temp = 0.0;

    for (int j = 0; j < c; j++) {
        if (dev_array[i * c + j] > temp) {
            temp = dev_array[i * c + j];
            idx = j;
        }
    }

    dev_argMax[i] = idx;
}

int main() {
    // Set the parameters
    const int r = 4; // Number of rows
    const int c = 5; // Number of columns

    // Host arrays
    float* h_array = (float*)malloc(r * c * sizeof(float));
    int* h_argMax = (int*)malloc(r * sizeof(int));

    // Initialize host arrays (example data, modify as needed)
    for (int i = 0; i < r * c; ++i) {
        h_array[i] = static_cast<float>(i % 10); // Example data, you can modify this accordingly
    }

    // Device arrays
    float* d_array;
    int* d_argMax;

    cudaMalloc((void**)&d_array, r * c * sizeof(float));
    cudaMalloc((void**)&d_argMax, r * sizeof(int));

    // Copy host data to device
    cudaMemcpy(d_array, h_array, r * c * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((r + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Launch the CUDA kernel
    Kernel_Argmax<<<blocksPerGrid, threadsPerBlock>>>(d_argMax, d_array, r, c);

    // Copy the result back to the host (optional)
    cudaMemcpy(h_argMax, d_argMax, r * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Argmax Result:\n");
    for (int i = 0; i < r; ++i) {
        printf("Row %d: Argmax=%d\n", i, h_argMax[i]);
    }

    // Clean up
    free(h_array);
    free(h_argMax);
    cudaFree(d_array);
    cudaFree(d_argMax);

    return 0;
}
 
