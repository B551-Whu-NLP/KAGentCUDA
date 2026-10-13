#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void Kernel_Avg(float* dev_arrayMax, float* dev_array, const int r, const int c) {
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int N = r;
    float sum;
    int i;

    while (tid < N) {
        i = tid;
        sum = 0.0;

        for (int j = 0; j < c; j++) {
            sum += dev_array[i * c + j];
        }

        dev_arrayMax[i] = sum / c;
        tid += gridDim.x * blockDim.x;
    }
}

int main() {
    // Set the dimensions of the array
    const int r = 100; // Change this according to your requirements
    const int c = 64;  // Change this according to your requirements

    // Host arrays
    float* h_array = (float*)malloc(r * c * sizeof(float));
    float* h_arrayMax = (float*)malloc(r * sizeof(float));

    // Initialize host arrays (example data, modify as needed)
    for (int i = 0; i < r * c; ++i) {
        h_array[i] = i; // Example data, you can modify this accordingly
    }

    // Device arrays
    float* d_array;
    float* d_arrayMax;

    cudaMalloc((void**)&d_array, r * c * sizeof(float));
    cudaMalloc((void**)&d_arrayMax, r * sizeof(float));

    // Copy host data to device
    cudaMemcpy(d_array, h_array, r * c * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int threadsPerBlock = 256; // Adjust this according to your requirements
    int blocksPerGrid = (r + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    Kernel_Avg<<<blocksPerGrid, threadsPerBlock>>>(d_arrayMax, d_array, r, c);

    // Copy the result back to the host (optional)
    cudaMemcpy(h_arrayMax, d_arrayMax, r * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Average for each row:\n");
    for (int i = 0; i < r; ++i) {
        printf("%.2f\t", h_arrayMax[i]);
    }
    printf("\n");

    // Clean up
    free(h_array);
    free(h_arrayMax);
    cudaFree(d_array);
    cudaFree(d_arrayMax);

    return 0;
}
 
