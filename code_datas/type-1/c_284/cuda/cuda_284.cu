#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void getMeanImage(const double* images, double* meanImage, int imageNum, int pixelNum) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= pixelNum) {
        return;
    }

    meanImage[col] = 0.0;

    for (int row = 0; row < imageNum; ++row) {
        meanImage[col] += images[row * pixelNum + col];
    }

    meanImage[col] /= imageNum;
}

int main() {
    // Set the dimensions of the images
    const int imageNum = 100; // Change this according to your requirements
    const int pixelNum = 64;  // Change this according to your requirements

    // Host arrays
    double* h_images = (double*)malloc(imageNum * pixelNum * sizeof(double));
    double* h_meanImage = (double*)malloc(pixelNum * sizeof(double));

    // Initialize host arrays (example data, modify as needed)
    for (int i = 0; i < imageNum * pixelNum; ++i) {
        h_images[i] = i; // Example data, you can modify this accordingly
    }

    // Device arrays
    double* d_images;
    double* d_meanImage;

    cudaMalloc((void**)&d_images, imageNum * pixelNum * sizeof(double));
    cudaMalloc((void**)&d_meanImage, pixelNum * sizeof(double));

    // Copy host data to device
    cudaMemcpy(d_images, h_images, imageNum * pixelNum * sizeof(double), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int threadsPerBlock = 256; // Adjust this according to your requirements
    int blocksPerGrid = (pixelNum + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    getMeanImage<<<blocksPerGrid, threadsPerBlock>>>(d_images, d_meanImage, imageNum, pixelNum);

    // Copy the result back to the host (optional)
    cudaMemcpy(h_meanImage, d_meanImage, pixelNum * sizeof(double), cudaMemcpyDeviceToHost);

    // Display the result (optional)
    printf("Mean Image:\n");
    for (int i = 0; i < pixelNum; ++i) {
        printf("%.2f\t", h_meanImage[i]);
    }
    printf("\n");

    // Clean up
    free(h_images);
    free(h_meanImage);
    cudaFree(d_images);
    cudaFree(d_meanImage);

    return 0;
}
 
