#include <stdio.h>
#include <cuda_runtime.h>

__global__ void distanceMatFinal(long int totalPixels, int availablePixels, int outPixelOffset, float *distMat) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (long int i = index; i < availablePixels; i += stride) {
        float sum = 0.0;
        float max = 0.0;

        for (long int j = 0; j < totalPixels; j++) {
            float element = distMat[i * totalPixels + j];
            if (element > max) max = element;
            sum += element;
        }

        sum += max;

        for (long int j = 0; j < totalPixels; j++) {
            if ((i + outPixelOffset) == j)
                distMat[i * totalPixels + j] = max / sum;
            else
                distMat[i * totalPixels + j] /= sum;
        }
    }
}

int main() {
    // Set array dimensions and other parameters
    const long int totalPixels = 100;   // Set the appropriate value
    const int availablePixels = 50;     // Set the appropriate value
    const int outPixelOffset = 10;      // Set the appropriate value

    // Allocate host memory
    float *distMat_host = (float *)malloc(totalPixels * availablePixels * sizeof(float));

    // Initialize input array (you may use your own initialization logic)
    // Note: You need to fill distMat_host with valid data

    // Allocate device memory
    float *distMat_device;
    cudaMalloc((void **)&distMat_device, totalPixels * availablePixels * sizeof(float));

    // Copy input array from host to device
    cudaMemcpy(distMat_device, distMat_host, totalPixels * availablePixels * sizeof(float), cudaMemcpyHostToDevice);

    // Set block and grid sizes
    dim3 blockSize(256);  // You may adjust the block size
    dim3 gridSize((availablePixels + blockSize.x - 1) / blockSize.x);

    // Launch the distanceMatFinal kernel
    distanceMatFinal<<<gridSize, blockSize>>>(totalPixels, availablePixels, outPixelOffset, distMat_device);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Copy the result array from device to host (if needed)
    cudaMemcpy(distMat_host, distMat_device, totalPixels * availablePixels * sizeof(float), cudaMemcpyDeviceToHost);

    // Print or process the result as needed

    // Cleanup
    cudaFree(distMat_device);
    free(distMat_host);

    return 0;
}
 
