#include <stdio.h>
#include <cuda_runtime.h>

__global__ void deInterleave_kernel2(float *d_X_out, float *d_Y_out, char *d_XY_in, int pitch_out, int pitch_in, int width, int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < width) & (y < height)) {
        float *data = (float *)(d_XY_in + y * pitch_in) + 2 * x;
        *((float *)((char *)d_X_out + y * pitch_out) + x) = data[0];
        *((float *)((char *)d_Y_out + y * pitch_out) + x) = data[1];
    }
}

int main() {
    // Set array dimensions
    const int width = 100;  // Set the appropriate value
    const int height = 100; // Set the appropriate value

    // Allocate host memory
    float *d_X_out_host = (float *)malloc(width * height * sizeof(float));
    float *d_Y_out_host = (float *)malloc(width * height * sizeof(float));
    char *d_XY_in_host = (char *)malloc(2 * width * height * sizeof(float)); // Assuming each element is a float

    // Initialize input array (you may use your own initialization logic)
    // Note: You need to fill d_XY_in_host with valid data

    // Allocate device memory
    float *d_X_out_device, *d_Y_out_device;
    char *d_XY_in_device;

    cudaMalloc((void **)&d_X_out_device, width * height * sizeof(float));
    cudaMalloc((void **)&d_Y_out_device, width * height * sizeof(float));
    cudaMalloc((void **)&d_XY_in_device, 2 * width * height * sizeof(float));

    // Copy input array from host to device
    cudaMemcpy(d_X_out_device, d_X_out_host, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y_out_device, d_Y_out_host, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_XY_in_device, d_XY_in_host, 2 * width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Set block and grid sizes
    dim3 blockSize(16, 16); // You may adjust the block size
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the deinterleave kernel
    deInterleave_kernel2<<<gridSize, blockSize>>>(d_X_out_device, d_Y_out_device, d_XY_in_device, 2 * width, 2 * width, width, height);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Copy the result array from device to host (if needed)
    cudaMemcpy(d_X_out_host, d_X_out_device, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_Y_out_host, d_Y_out_device, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Print or process the result as needed

    // Cleanup
    cudaFree(d_X_out_device);
    cudaFree(d_Y_out_device);
    cudaFree(d_XY_in_device);
    free(d_X_out_host);
    free(d_Y_out_host);
    free(d_XY_in_host);

    return 0;
}
 
