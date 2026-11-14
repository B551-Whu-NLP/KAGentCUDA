#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void equalization(float *cdf, float *mincdf, unsigned char *ucharImage, int imageWidth, int imageHeight, int channels, int pixelSize) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < pixelSize) {
        unsigned char val = ucharImage[idx];
        float data = 255 * (cdf[val] - mincdf[0]) / (1 - mincdf[0]);
        if (data < 0.0f)
            data = 0.0f;
        else if (data > 255.0f)
            data = 255.0f;
        ucharImage[idx] = (unsigned char)data;
    }
}

int main() {
    // Set your image dimensions and other parameters
    const int imageWidth = 512;
    const int imageHeight = 512;
    const int channels = 3; // Assuming RGB image
    const int pixelSize = imageWidth * imageHeight * channels;

    // Allocate host memory
    float *h_cdf = (float *)malloc(256 * sizeof(float));
    float *h_mincdf = (float *)malloc(sizeof(float));
    unsigned char *h_ucharImage = (unsigned char *)malloc(pixelSize * sizeof(unsigned char));

    // Initialize host data (replace with your data initialization logic)
    for (int i = 0; i < 256; ++i) {
        h_cdf[i] = static_cast<float>(i);
    }

    *h_mincdf = 0.5f; // Replace with the actual value

    for (int i = 0; i < pixelSize; ++i) {
        h_ucharImage[i] = static_cast<unsigned char>(i % 256);
    }

    // Allocate device memory
    float *d_cdf, *d_mincdf;
    unsigned char *d_ucharImage;
    cudaMalloc((void **)&d_cdf, 256 * sizeof(float));
    cudaMalloc((void **)&d_mincdf, sizeof(float));
    cudaMalloc((void **)&d_ucharImage, pixelSize * sizeof(unsigned char));

    // Copy data from host to device
    cudaMemcpy(d_cdf, h_cdf, 256 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mincdf, h_mincdf, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ucharImage, h_ucharImage, pixelSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 blockSize(256); // You may adjust the block size
    dim3 gridSize((pixelSize + blockSize.x - 1) / blockSize.x);

    // Launch the kernel
    equalization<<<gridSize, blockSize>>>(d_cdf, d_mincdf, d_ucharImage, imageWidth, imageHeight, channels, pixelSize);

    // Copy result back to host (optional, depends on your application)
    cudaMemcpy(h_ucharImage, d_ucharImage, pixelSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Print or process the result as needed
    printf("Results printed here:\n");

    // Cleanup
    free(h_cdf);
    free(h_mincdf);
    free(h_ucharImage);
    cudaFree(d_cdf);
    cudaFree(d_mincdf);
    cudaFree(d_ucharImage);

    return 0;
}
 
