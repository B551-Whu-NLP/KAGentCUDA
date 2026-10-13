#include <stdio.h>

__global__ void subtractMean(double* images, const double* meanImage, size_t imageNum, size_t pixelNum) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= pixelNum) {
        return;
    }

    for (size_t row = 0; row < imageNum; ++row) {
        images[row * pixelNum + col] -= meanImage[col];

        if (images[row * pixelNum + col] < 0.0) {
            images[row * pixelNum + col] = 0.0;
        }
    }
}

int main() {
    // Set your desired parameters
    size_t imageNum = 512;   // Set your desired value for imageNum
    size_t pixelNum = 1024;  // Set your desired value for pixelNum

    // Allocate memory on the host
    double* h_images = (double*)malloc(imageNum * pixelNum * sizeof(double));
    double* h_meanImage = (double*)malloc(pixelNum * sizeof(double));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    double* d_images, * d_meanImage;
    cudaMalloc((void**)&d_images, imageNum * pixelNum * sizeof(double));
    cudaMalloc((void**)&d_meanImage, pixelNum * sizeof(double));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((pixelNum + 255) / 256, 1, 1);
    dim3 blockSize(256, 1, 1);

    // Launch the CUDA kernel for subtracting mean
    subtractMean<<<gridSize, blockSize>>>(d_images, d_meanImage, imageNum, pixelNum);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_images);
    cudaFree(d_meanImage);

    // Free host memory
    free(h_images);
    free(h_meanImage);

    return 0;
}
