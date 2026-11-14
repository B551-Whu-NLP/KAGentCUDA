#include <stdio.h>

// CUDA kernel function
__global__ void transKernel(float* array1, float* array2, int width) {
    int current_index =
        (blockIdx.y * blockDim.y + threadIdx.y) * width +
        (blockIdx.x * blockDim.x + threadIdx.x);
    int replace =
        (blockIdx.x * blockDim.x + threadIdx.x) * width +
        blockIdx.y * blockDim.y + threadIdx.y;

    if (current_index < width * width) {
        array2[replace] = array1[current_index];
    }
}

int main() {
    // Matrix size
    int width = 4;  // Change this according to your requirements

    // Host arrays
    float* h_array1 = (float*)malloc(width * width * sizeof(float));
    float* h_array2 = (float*)malloc(width * width * sizeof(float));

    // Initialize host input arrays
    for (int i = 0; i < width * width; ++i) {
        h_array1[i] = i;  // Example data for array1, you can modify this accordingly
    }

    // Device arrays
    float* d_array1;
    float* d_array2;
    cudaMalloc((void**)&d_array1, width * width * sizeof(float));
    cudaMalloc((void**)&d_array2, width * width * sizeof(float));

    // Copy host input arrays to device
    cudaMemcpy(d_array1, h_array1, width * width * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block_size(2, 2);  // Change this according to your requirements
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (width + block_size.y - 1) / block_size.y);

    // Launch the CUDA kernel
    transKernel<<<grid_size, block_size>>>(d_array1, d_array2, width);

    // Copy the result back to the host
    cudaMemcpy(h_array2, d_array2, width * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Results:\n");
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%f ", h_array2[i * width + j]);
        }
        printf("\n");
    }

    // Clean up
    free(h_array1);
    free(h_array2);
    cudaFree(d_array1);
    cudaFree(d_array2);

    return 0;
}
 
