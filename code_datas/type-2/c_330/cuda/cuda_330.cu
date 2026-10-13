#include <stdio.h>
#include <cuda_runtime.h>

__global__ void Kernel_Softmax(float *dev_x, const int r, const int c) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= r)
        return;

    float temp1 = 0., temp2 = 0.;
    for (int j = 0; j < c; j++)
        temp1 = fmaxf(dev_x[i * c + j], temp1);

    for (int j = 0; j < c; j++) {
        dev_x[i * c + j] = expf(dev_x[i * c + j] - temp1);
        temp2 += dev_x[i * c + j];
    }

    for (int j = 0; j < c; j++)
        dev_x[i * c + j] /= temp2;
}

int main() {
    // Set array dimensions
    const int r = 100;  // Set the appropriate value
    const int c = 10;   // Set the appropriate value

    // Allocate host memory
    float *dev_x_host = (float *)malloc(r * c * sizeof(float));

    // Initialize input array (you may use your own initialization logic)
    // Note: You need to fill dev_x_host with valid data

    // Allocate device memory
    float *dev_x_device;
    cudaMalloc((void **)&dev_x_device, r * c * sizeof(float));

    // Copy input array from host to device
    cudaMemcpy(dev_x_device, dev_x_host, r * c * sizeof(float), cudaMemcpyHostToDevice);

    // Set block and grid sizes
    dim3 blockSize(256);  // You may adjust the block size
    dim3 gridSize((r + blockSize.x - 1) / blockSize.x);

    // Launch the softmax kernel
    Kernel_Softmax<<<gridSize, blockSize>>>(dev_x_device, r, c);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Copy the result array from device to host (if needed)
    cudaMemcpy(dev_x_host, dev_x_device, r * c * sizeof(float), cudaMemcpyDeviceToHost);

    // Print or process the result as needed

    // Cleanup
    cudaFree(dev_x_device);
    free(dev_x_host);

    return 0;
}
 
