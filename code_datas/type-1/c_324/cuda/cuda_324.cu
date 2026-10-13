#include <stdio.h>
#include <cuda_runtime.h>

__global__ void castImageToGrayScale(unsigned char *ucharImage, unsigned char *grayImage, int imageWidth, int imageHeight, int channels) {
    int w = threadIdx.x + blockDim.x * blockIdx.x;
    int h = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = imageWidth * h + w;

    if (w < imageWidth && h < imageHeight) {
        unsigned char r = ucharImage[idx * channels];
        unsigned char g = ucharImage[idx * channels + 1];
        unsigned char b = ucharImage[idx * channels + 2];

        grayImage[idx] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

int main() {
    // Set image dimensions and channels
    int imageWidth = 512;   // Set the appropriate value
    int imageHeight = 512;  // Set the appropriate value
    int channels = 3;       // Set the appropriate value

    // Allocate host memory
    unsigned char *h_ucharImage, *h_grayImage;
    h_ucharImage = (unsigned char *)malloc(imageWidth * imageHeight * channels * sizeof(unsigned char));
    h_grayImage = (unsigned char *)malloc(imageWidth * imageHeight * sizeof(unsigned char));

    // Initialize ucharImage array (you may use your own initialization logic)
    for (int i = 0; i < imageWidth * imageHeight * channels; i++) {
        h_ucharImage[i] = (unsigned char)(i % 256);  // Example: Simple initialization
    }

    // Allocate device memory
    unsigned char *d_ucharImage, *d_grayImage;
    cudaMalloc((void **)&d_ucharImage, imageWidth * imageHeight * channels * sizeof(unsigned char));
    cudaMalloc((void **)&d_grayImage, imageWidth * imageHeight * sizeof(unsigned char));

    // Copy ucharImage array from host to device
    cudaMemcpy(d_ucharImage, h_ucharImage, imageWidth * imageHeight * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 blockSize(16, 16);  // You may adjust the block size
    dim3 gridSize((imageWidth + blockSize.x - 1) / blockSize.x, (imageHeight + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    castImageToGrayScale<<<gridSize, blockSize>>>(d_ucharImage, d_grayImage, imageWidth, imageHeight, channels);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Check for errors during the kernel launch
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    // Copy the result array from device to host
    cudaMemcpy(h_grayImage, d_grayImage, imageWidth * imageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Display the result array (you may modify this part based on your needs)
    printf("Gray Image:\n");
    for (int i = 0; i < imageHeight; i++) {
        for (int j = 0; j < imageWidth; j++) {
            printf("%u ", h_grayImage[i * imageWidth + j]);
        }
        printf("\n");
    }

    // Cleanup
    free(h_ucharImage);
    free(h_grayImage);
    cudaFree(d_ucharImage);
    cudaFree(d_grayImage);

    return 0;
}
 
