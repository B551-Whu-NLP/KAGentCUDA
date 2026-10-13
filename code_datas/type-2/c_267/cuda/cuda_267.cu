#include <stdio.h>

// CUDA kernel function
__global__ void flipKernel(float* array1, int width) {
    int current_index = blockIdx.x * blockDim.x + threadIdx.x;
    int replace = (width - 1 - current_index / width) * width + current_index % width;

    if (current_index < width * width / 2) {
        float temp = array1[current_index];
        array1[current_index] = array1[replace];
        array1[replace] = temp;
    }
}

int main() {
    // Array size (assuming a square matrix)
    int width = 4; // Change this according to your requirements

    // Host array
    float* h_array1 = (float*)malloc(width * width * sizeof(float));

    // Initialize host input array
    for (int i = 0; i < width * width; ++i) {
        h_array1[i] = i + 1.0; // Example data for array1, you can modify this accordingly
    }

    // Device array
    float* d_array1;
    cudaMalloc((void**)&d_array1, width * width * sizeof(float));

    // Copy host input array to device
    cudaMemcpy(d_array1, h_array1, width * width * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((width * width + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    flipKernel<<<grid_size, block_size>>>(d_array1, width);

    // Copy the result back to the host
    cudaMemcpy(h_array1, d_array1, width * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Original Array:\n");
    for (int i = 0; i < width * width; ++i) {
        printf("%.2f ", h_array1[i]);
        if ((i + 1) % width == 0) {
            printf("\n");
        }
    }

    // Clean up
    free(h_array1);
    cudaFree(d_array1);

    return 0;
}
 
