#include <stdio.h>

__global__ void vectorMatrixMult(long int totalPixels, int availablePixels, int outPixelOffset, float* matrix, float* vector, float* out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (long int i = index; i < availablePixels; i += stride) {
        float sum = 0.0;

        for (long int j = 0; j < totalPixels; j++) {
            sum += matrix[i * totalPixels + j] * vector[j];
        }

        out[i + outPixelOffset] = sum;
    }
}

int main() {
    // Set your desired parameters
    long int totalPixels = 1024;      // Set your desired value for totalPixels
    int availablePixels = 512;        // Set your desired value for availablePixels
    int outPixelOffset = 256;         // Set your desired value for outPixelOffset

    // Allocate memory on the host
    float* h_matrix = (float*)malloc(availablePixels * totalPixels * sizeof(float));
    float* h_vector = (float*)malloc(totalPixels * sizeof(float));
    float* h_out = (float*)malloc(availablePixels * sizeof(float));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    float* d_matrix, * d_vector, * d_out;
    cudaMalloc((void**)&d_matrix, availablePixels * totalPixels * sizeof(float));
    cudaMalloc((void**)&d_vector, totalPixels * sizeof(float));
    cudaMalloc((void**)&d_out, availablePixels * sizeof(float));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((availablePixels + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);

    // Launch the CUDA kernel for vector-matrix multiplication
    vectorMatrixMult<<<gridSize, blockSize>>>(totalPixels, availablePixels, outPixelOffset, d_matrix, d_vector, d_out);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_out);

    // Free host memory
    free(h_matrix);
    free(h_vector);
    free(h_out);

    return 0;
}
