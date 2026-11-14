#include <stdio.h>

// CUDA kernel function
__global__ void castImageToUchar(float* deviceInputImageData, unsigned char* ucharImage,
                                  int imageWidth, int imageHeight, int channels, int pixelSize) {
    int w = threadIdx.x + blockDim.x * blockIdx.x;

    if (w < pixelSize) {
        ucharImage[w] = static_cast<unsigned char>(255 * deviceInputImageData[w]);
    }
}

int main() {
    // Image parameters
    int imageWidth = 512;    // Change this according to your requirements
    int imageHeight = 512;   // Change this according to your requirements
    int channels = 3;        // Change this according to your requirements
    int pixelSize = channels * imageWidth * imageHeight;

    // Host arrays
    float* h_deviceInputImageData = (float*)malloc(pixelSize * sizeof(float));
    unsigned char* h_ucharImage = (unsigned char*)malloc(pixelSize * sizeof(unsigned char));

    // Initialize host input arrays
    for (int i = 0; i < pixelSize; ++i) {
        h_deviceInputImageData[i] = static_cast<float>(i % 256) / 255.0f;  // Example data, you can modify this accordingly
    }

    // Device arrays
    float* d_deviceInputImageData;
    unsigned char* d_ucharImage;
    cudaMalloc((void**)&d_deviceInputImageData, pixelSize * sizeof(float));
    cudaMalloc((void**)&d_ucharImage, pixelSize * sizeof(unsigned char));

    // Copy host input arrays to device
    cudaMemcpy(d_deviceInputImageData, h_deviceInputImageData, pixelSize * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((pixelSize + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    castImageToUchar<<<grid_size, block_size>>>(d_deviceInputImageData, d_ucharImage,
                                                 imageWidth, imageHeight, channels, pixelSize);

    // Copy the result back to the host
    cudaMemcpy(h_ucharImage, d_ucharImage, pixelSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Result: ");
    for (int i = 0; i < pixelSize; ++i) {
        printf("%d ", h_ucharImage[i]);
    }
    printf("\n");

    // Clean up
    free(h_deviceInputImageData);
    free(h_ucharImage);
    cudaFree(d_deviceInputImageData);
    cudaFree(d_ucharImage);

    return 0;
}
 
