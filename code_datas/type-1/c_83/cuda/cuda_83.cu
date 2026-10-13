#include <stdio.h>

// CUDA kernel for converting float to RGBA
__global__ void convertFloatToRGBA_kernel(char* out_image, const float* in_image, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int IND = y * width + x;
        float val = in_image[IND];
        char temp = 255;
        out_image[IND] = temp;
    }
}

int main() {
    // Set your desired image dimensions
    int width = 512;
    int height = 512;

    // Allocate memory on the host
    char* h_out_image = (char*)malloc(width * height * sizeof(char));
    float* h_in_image = (float*)malloc(width * height * sizeof(float));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    char* d_out_image;
    float* d_in_image;
    cudaMalloc((void**)&d_out_image, width * height * sizeof(char));
    cudaMalloc((void**)&d_in_image, width * height * sizeof(float));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    dim3 blockSize(16, 16);

    // Launch the CUDA kernel for converting float to RGBA
    convertFloatToRGBA_kernel<<<gridSize, blockSize>>>(d_out_image, d_in_image, width, height);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_out_image);
    cudaFree(d_in_image);

    // Free host memory
    free(h_out_image);
    free(h_in_image);

    return 0;
}
