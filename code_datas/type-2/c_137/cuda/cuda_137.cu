#include <stdio.h>
#include <math.h>

// CUDA kernel
__global__ void variance_kernel(float* x, float* mean, int batch, int filters, int spatial, float* variance) {
    float scale = 1.f / (batch * spatial - 1);
    int j, k;
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i >= filters)
        return;

    variance[i] = 0;

    for (j = 0; j < batch; ++j) {
        for (k = 0; k < spatial; ++k) {
            int index = j * filters * spatial + i * spatial + k;
            variance[i] += powf((x[index] - mean[i]), 2);
        }
    }

    variance[i] *= scale;
}

int main() {
    // Your main program logic here

    // Example: Allocate and initialize host and device memory
    int batch = 32;     // Replace with your actual batch size
    int filters = 64;   // Replace with your actual number of filters
    int spatial = 128;  // Replace with your actual spatial size

    float* h_x = (float*)malloc(batch * filters * spatial * sizeof(float));
    float* h_mean = (float*)malloc(filters * sizeof(float));
    float* h_variance = (float*)malloc(filters * sizeof(float));

    float* d_x, * d_mean, * d_variance;
    cudaMalloc((void**)&d_x, batch * filters * spatial * sizeof(float));
    cudaMalloc((void**)&d_mean, filters * sizeof(float));
    cudaMalloc((void**)&d_variance, filters * sizeof(float));

    // Copy host memory to device
    cudaMemcpy(d_x, h_x, batch * filters * spatial * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, h_mean, filters * sizeof(float), cudaMemcpyHostToDevice);

    // Configure and launch the CUDA kernel
    dim3 blockSize(256); // Adjust block dimensions based on your requirements
    dim3 gridSize((filters + blockSize.x - 1) / blockSize.x, 1);

    variance_kernel<<<gridSize, blockSize>>>(d_x, d_mean, batch, filters, spatial, d_variance);

    // Copy device memory back to host
    cudaMemcpy(h_variance, d_variance, filters * sizeof(float), cudaMemcpyDeviceToHost);

    // Your post-kernel logic here

    // Free allocated memory
    free(h_x);
    free(h_mean);
    free(h_variance);
    cudaFree(d_x);
    cudaFree(d_mean);
    cudaFree(d_variance);

    return 0;
}
