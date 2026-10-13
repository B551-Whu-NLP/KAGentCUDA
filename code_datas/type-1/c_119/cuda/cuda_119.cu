#include <stdio.h>

// CUDA kernel
__global__ void colorConvert(unsigned char* grayImage, unsigned char* colorImage, int rows, int columns) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if ((column < columns) && (row < rows)) {
        int offset = column + (columns * row);
        unsigned char grayValue = 0.07 * colorImage[offset * 3] + 0.71 * colorImage[offset * 3 + 1] + 0.21 * colorImage[offset * 3 + 2];
        grayImage[offset] = grayValue;
    }
}

int main() {
    // Your main program logic here

    // Example: Allocate and initialize host and device memory
    int rows = 100; // Replace with your actual dimensions
    int columns = 100;

    unsigned char* h_grayImage = (unsigned char*)malloc(rows * columns * sizeof(unsigned char));
    unsigned char* h_colorImage = (unsigned char*)malloc(rows * columns * 3 * sizeof(unsigned char));

    unsigned char* d_grayImage, * d_colorImage;
    cudaMalloc((void**)&d_grayImage, rows * columns * sizeof(unsigned char));
    cudaMalloc((void**)&d_colorImage, rows * columns * 3 * sizeof(unsigned char));

    // Copy host memory to device
    cudaMemcpy(d_grayImage, h_grayImage, rows * columns * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colorImage, h_colorImage, rows * columns * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Configure and launch the CUDA kernel
    dim3 blockSize(16, 16); // Adjust block dimensions based on your requirements
    dim3 gridSize((columns + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
    colorConvert<<<gridSize, blockSize>>>(d_grayImage, d_colorImage, rows, columns);

    // Copy device memory back to host
    cudaMemcpy(h_grayImage, d_grayImage, rows * columns * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Your post-kernel logic here

    // Free allocated memory
    free(h_grayImage);
    free(h_colorImage);
    cudaFree(d_grayImage);
    cudaFree(d_colorImage);

    return 0;
}
