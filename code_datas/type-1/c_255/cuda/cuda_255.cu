#include <stdio.h>

// CUDA kernel function
__global__ void castImageTofloat(float* deviceOutputImageData, unsigned char* ucharImage,
                                  int imageWidth, int imageHeight, int channels, int pixelSize) {
    int w = threadIdx.x + blockDim.x * blockIdx.x;

    if (w < pixelSize) {
        deviceOutputImageData[w] = static_cast<float>(ucharImage[w]) / 255.0f;
    }
}

int main() {
    // Image parameters
    int imageWidth = 512;    // Change this according to your requirements
    int imageHeight = 512;   // Change this according to your requirements
    int channels = 3;        // Change this according to your requirements
    int pixelSize = channels * imageWidth * imageHeight;

    // Host arrays
    float* h_deviceOutputImageData = (float*)malloc(pixelSize * sizeof(float));
    unsigned char* h_ucharImage = (unsigned char*)malloc(pixelSize * sizeof(unsigned char));

    // Initialize host input arrays
    for (int i = 0; i < pixelSize; ++i) {
        h_ucharImage[i] = static_cast<unsigned char>(i % 256);  // Example data, you can modify this accordingly
    }

    // Device arrays
    float* d_deviceOutputImageData;
    unsigned char* d_ucharImage;
    cudaMalloc((void**)&d_deviceOutputImageData, pixelSize * sizeof(float));
    cudaMalloc((void**)&d_ucharImage, pixelSize * sizeof(unsigned char));

    // Copy host input arrays to device
    cudaMemcpy(d_ucharImage, h_ucharImage, pixelSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int block_size = 256;
    dim3 grid_size((pixelSize + block_size - 1) / block_size, 1);

    // Launch the CUDA kernel
    castImageTofloat<<<grid_size, block_size>>>(d_deviceOutputImageData, d_ucharImage,
                                               imageWidth, imageHeight, channels, pixelSize);

    // Copy the result back to the host
    cudaMemcpy(h_deviceOutputImageData, d_deviceOutputImageData, pixelSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result
    printf("Result: ");
    for (int i = 0; i < pixelSize; ++i) {
        printf("%f ", h_deviceOutputImageData[i]);
    }
    printf("\n");

    // Clean up
    free(h_deviceOutputImageData);
    free(h_ucharImage);
    cudaFree(d_deviceOutputImageData);
    cudaFree(d_ucharImage);

    return 0;
}
 
