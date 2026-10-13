#include <stdio.h>

__global__ void convertKinectDisparityInPlace_kernel(float* d_disparity, int pitch, int width, int height, float depth_scale) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < width) && (y < height)) {
        float* d_in = (float*)((char*)d_disparity + y * pitch) + x;
        *d_in = (*d_in == 0.0f) ? 1 : (-depth_scale / *d_in);
    }
}

int main() {
    // Set your desired parameters
    int width = 640;   // Set your desired value for width
    int height = 480;  // Set your desired value for height
    float depth_scale = 0.001f;  // Set your desired value for depth_scale

    // Allocate memory on the host
    float* h_disparity = (float*)malloc(width * height * sizeof(float));

    // Initialize or copy data to host memory

    // Allocate memory on the device
    float* d_disparity;
    cudaMalloc((void**)&d_disparity, width * height * sizeof(float));

    // Copy data to device memory

    // Calculate grid and block dimensions
    dim3 gridSize((width + 15) / 16, (height + 15) / 16, 1);
    dim3 blockSize(16, 16, 1);

    // Launch the CUDA kernel for converting Kinect disparity in place
    convertKinectDisparityInPlace_kernel<<<gridSize, blockSize>>>(d_disparity, width * sizeof(float), width, height, depth_scale);

    // Copy the result back to the host

    // Free device memory
    cudaFree(d_disparity);

    // Free host memory
    free(h_disparity);

    return 0;
}
