#include <stdio.h>
#include <cuda_runtime.h>

__global__ void cuda_laplace_filter(float *Img, float *laplace, float _dz, float _dx, int npml, int nnz, int nnx) {
    int i1 = threadIdx.x + blockDim.x * blockIdx.x;
    int i2 = threadIdx.y + blockDim.y * blockIdx.y;
    int id = i1 + i2 * nnz;
    float diff1 = 0.0f;
    float diff2 = 0.0f;

    if (i1 >= npml && i1 < nnz - npml && i2 >= npml && i2 < nnx - npml) {
        diff1 = Img[id + 1] - 2.0 * Img[id] + Img[id - 1];
        diff2 = Img[id + nnz] - 2.0 * Img[id] + Img[id - nnz];
    }

    laplace[id] = _dz * _dz * diff1 + _dx * _dx * diff2;
}

int main() {
    // Set array dimensions and other parameters
    const int nnz = 100;  // Set the appropriate value
    const int nnx = 100;  // Set the appropriate value
    const int npml = 5;   // Set the appropriate value

    // Allocate host memory
    float *Img_host = (float *)malloc(nnz * nnx * sizeof(float));
    float *laplace_host = (float *)malloc(nnz * nnx * sizeof(float));

    // Initialize input array (you may use your own initialization logic)
    // Note: You need to fill Img_host with valid data

    // Allocate device memory
    float *Img_device, *laplace_device;

    cudaMalloc((void **)&Img_device, nnz * nnx * sizeof(float));
    cudaMalloc((void **)&laplace_device, nnz * nnx * sizeof(float));

    // Copy input array from host to device
    cudaMemcpy(Img_device, Img_host, nnz * nnx * sizeof(float), cudaMemcpyHostToDevice);

    // Set block and grid sizes
    dim3 blockSize(16, 16); // You may adjust the block size
    dim3 gridSize((nnz + blockSize.x - 1) / blockSize.x, (nnx + blockSize.y - 1) / blockSize.y);

    // Launch the Laplace filter kernel
    cuda_laplace_filter<<<gridSize, blockSize>>>(Img_device, laplace_device, 1.0, 1.0, npml, nnz, nnx);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Copy the result array from device to host (if needed)
    cudaMemcpy(laplace_host, laplace_device, nnz * nnx * sizeof(float), cudaMemcpyDeviceToHost);

    // Print or process the result as needed

    // Cleanup
    cudaFree(Img_device);
    cudaFree(laplace_device);
    free(Img_host);
    free(laplace_host);

    return 0;
}
 
