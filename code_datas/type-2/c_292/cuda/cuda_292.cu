#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void histogrammPrimitive(unsigned int* histogrammVector, unsigned char* grayImage, int rows, int columns) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = (column) + (columns * row);

    if ((column < columns) && (row < rows)) {
        unsigned char grayValue = grayImage[offset];
        atomicAdd(&(histogrammVector[grayValue]), 1);
    }
}

int main() {
    // Set the parameters
    const int rows = 512; // Change as needed
    const int columns = 512; // Change as needed
    const int histogramSize = 256; // Assuming an 8-bit image

    // Host arrays
    unsigned char* h_grayImage = (unsigned char*)malloc(rows * columns * sizeof(unsigned char));
    unsigned int* h_histogramVector = (unsigned int*)malloc(histogramSize * sizeof(unsigned int));

    // Initialize host arrays (example data, modify as needed)
    for (int i = 0; i < rows * columns; ++i) {
        h_grayImage[i] = i % 256; // Example data, you can modify this accordingly
    }

    // Device arrays
    unsigned char* d_grayImage;
    unsigned int* d_histogramVector;

    cudaMalloc((void**)&d_grayImage, rows * columns * sizeof(unsigned char));
    cudaMalloc((void**)&d_histogramVector, histogramSize * sizeof(unsigned int));

    // Copy host data to device
    cudaMemcpy(d_grayImage, h_grayImage, rows * columns * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_histogramVector, 0, histogramSize * sizeof(unsigned int));

    // Define block and grid dimensions
    dim3 threadsPerBlock(16, 16); // 16x16 block size
    dim3 blocksPerGrid((columns + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the CUDA kernel
    histogrammPrimitive<<<blocksPerGrid, threadsPerBlock>>>(d_histogramVector, d_grayImage, rows, columns);

    // Copy the result back to the host (optional)
    cudaMemcpy(h_histogramVector, d_histogramVector, histogramSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Histogram:\n");
    for (int i = 0; i < histogramSize; ++i) {
        printf("Value %d: %u\n", i, h_histogramVector[i]);
    }

    // Clean up
    free(h_grayImage);
    free(h_histogramVector);
    cudaFree(d_grayImage);
    cudaFree(d_histogramVector);

    return 0;
}
 
