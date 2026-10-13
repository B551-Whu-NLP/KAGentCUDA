#include <stdio.h>
#include <cuda_runtime.h>

__global__ void flatten_kernel(int N, float *x, int spatial, int layers, int batch, int forward, float *out) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    int in_s = i % spatial;
    i = i / spatial;
    int in_c = i % layers;
    i = i / layers;
    int b = i;

    int i1 = b * layers * spatial + in_c * spatial + in_s;
    int i2 = b * layers * spatial + in_s * layers + in_c;

    if (forward)
        out[i2] = x[i1];
    else
        out[i1] = x[i2];
}

int main() {
    // Set array dimensions
    const int N = 1000;  // Set the appropriate value
    const int spatial = 10;  // Set the appropriate value
    const int layers = 5;  // Set the appropriate value
    const int batch = 2;  // Set the appropriate value

    // Allocate host memory
    float *x_host = (float *)malloc(N * sizeof(float));
    float *out_host = (float *)malloc(N * sizeof(float));

    // Initialize input arrays (you may use your own initialization logic)
    // Note: You need to fill x_host with valid data

    // Allocate device memory
    float *x_device, *out_device;
    cudaMalloc((void **)&x_device, N * sizeof(float));
    cudaMalloc((void **)&out_device, N * sizeof(float));

    // Copy input arrays from host to device
    cudaMemcpy(x_device, x_host, N * sizeof(float), cudaMemcpyHostToDevice);

    // Set block and grid sizes
    dim3 blockSize(256);  // You may adjust the block size
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    // Launch the kernel for forward operation
    flatten_kernel<<<gridSize, blockSize>>>(N, x_device, spatial, layers, batch, 1, out_device);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Copy the result array from device to host (if needed)
    cudaMemcpy(out_host, out_device, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print or process the result as needed

    // Cleanup
    cudaFree(x_device);
    cudaFree(out_device);
    free(x_host);
    free(out_host);

    return 0;
}
 
