#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void kernel_softmax(float *x, int r, int c) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= r)
        return;

    float temp1 = 0., temp2 = 0.;
    for (int j = 0; j < c; j++)
        temp1 = fmaxf(x[i * c + j], temp1);

    for (int j = 0; j < c; j++) {
        x[i * c + j] = expf(x[i * c + j] - temp1);
        temp2 += x[i * c + j];
    }

    for (int j = 0; j < c; j++)
        x[i * c + j] /= temp2;
}

int main() {
    // Set array dimensions
    const int r = 1000;  // Set the appropriate value
    const int c = 10;    // Set the appropriate value

    // Allocate host memory
    float *x_host = (float *)malloc(r * c * sizeof(float));

    // Initialize input array (you may use your own initialization logic)
    // Note: You need to fill x_host with valid data

    // Allocate device memory
    float *x_device;
    cudaMalloc((void **)&x_device, r * c * sizeof(float));

    // Copy input array from host to device
    cudaMemcpy(x_device, x_host, r * c * sizeof(float), cudaMemcpyHostToDevice);

    // Set block and grid sizes
    dim3 blockSize(256);  // You may adjust the block size
    dim3 gridSize((r + blockSize.x - 1) / blockSize.x);

    // Launch the kernel
    kernel_softmax<<<gridSize, blockSize>>>(x_device, r, c);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Check for errors during the kernel launch
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    // Copy the result array from device to host (if needed)

    // Cleanup
    cudaFree(x_device);
    free(x_host);

    return 0;
}
 
