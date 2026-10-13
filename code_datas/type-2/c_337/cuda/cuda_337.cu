#include <iostream>
#include <cmath>

// Define CUDA kernel
__global__ void Kernel_Softmax_seg(float *dev_x, const int c, const int size) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    int N = size;

    while (i < N) {
        float temp = 0.0;

        // Find max value
        for (int j = 0; j < c; j++) {
            temp = fmaxf(dev_x[j * size + i], temp);
        }

        // Subtract max and exponentiate
        for (int j = 0; j < c; j++) {
            dev_x[j * size + i] = expf(dev_x[j * size + i] - temp);
        }

        // Sum of exponentials
        temp = 0.0;
        for (int j = 0; j < c; j++) {
            temp += dev_x[j * size + i];
        }

        // Normalize
        for (int j = 0; j < c; j++) {
            dev_x[j * size + i] /= temp;
        }

        i += gridDim.x * blockDim.x;
    }
}

int main() {
    // Example usage
    int c = 5;  // Number of classes
    int size = 1000;  // Size of data
    int total_elements = c * size;

    // Allocate device memory
    float *d_dev_x;
    cudaMalloc((void**)&d_dev_x, total_elements * sizeof(float));

    // Initialize data on host (you should replace this with your actual data)
    float *h_dev_x = new float[total_elements];

    // Copy data from host to device
    cudaMemcpy(d_dev_x, h_dev_x, total_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x);

    // Launch the CUDA kernel
    Kernel_Softmax_seg<<<gridSize, blockSize>>>(d_dev_x, c, size);

    // Copy the result back to host
    cudaMemcpy(h_dev_x, d_dev_x, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory
    delete[] h_dev_x;
    cudaFree(d_dev_x);

    return 0;
}
 
