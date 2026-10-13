#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void vectorMatrixMult(long int totalPixels, float* matrix, float* vector, float* out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (long int i = index; i < totalPixels; i += stride) {
        float sum = 0.0;

        for (long int j = 0; j < totalPixels; j++) {
            sum += matrix[i * totalPixels + j] * vector[j];
        }

        out[i] = sum;
    }
}

int main() {
    // Set the parameters
    const long int totalPixels = 100; // Change this according to your requirements

    // Host arrays
    float* h_matrix = (float*)malloc(totalPixels * totalPixels * sizeof(float));
    float* h_vector = (float*)malloc(totalPixels * sizeof(float));
    float* h_out = (float*)malloc(totalPixels * sizeof(float));

    // Initialize host arrays (example data, modify as needed)
    for (long int i = 0; i < totalPixels * totalPixels; ++i) {
        h_matrix[i] = i; // Example data, you can modify this accordingly
    }

    for (long int i = 0; i < totalPixels; ++i) {
        h_vector[i] = i; // Example data, you can modify this accordingly
    }

    // Device arrays
    float* d_matrix;
    float* d_vector;
    float* d_out;

    cudaMalloc((void**)&d_matrix, totalPixels * totalPixels * sizeof(float));
    cudaMalloc((void**)&d_vector, totalPixels * sizeof(float));
    cudaMalloc((void**)&d_out, totalPixels * sizeof(float));

    // Copy host data to device
    cudaMemcpy(d_matrix, h_matrix, totalPixels * totalPixels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, totalPixels * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int threadsPerBlock = 256; // Adjust this according to your requirements
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    vectorMatrixMult<<<blocksPerGrid, threadsPerBlock>>>(totalPixels, d_matrix, d_vector, d_out);

    // Copy the result back to the host (optional)
    cudaMemcpy(h_out, d_out, totalPixels * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Result array:\n");
    for (long int i = 0; i < totalPixels; ++i) {
        printf("%f\t", h_out[i]);
    }
    printf("\n");

    // Clean up
    free(h_matrix);
    free(h_vector);
    free(h_out);
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_out);

    return 0;
}
 
