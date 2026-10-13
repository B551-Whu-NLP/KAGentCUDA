#include <stdio.h>

// Define the CUDA kernel
__global__ void upsample_kernel(size_t N, float *x, int w, int h, int c, int batch, int stride, int forward, float scale, float *out) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (i >= N)
        return;

    int out_index = i;
    int out_w = i % (w * stride);
    i = i / (w * stride);
    int out_h = i % (h * stride);
    i = i / (h * stride);
    int out_c = i % c;
    i = i / c;
    int b = i % batch;
    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;
    int in_index = b * w * h * c + in_c * w * h + in_h * w + in_w;

    if (forward)
        atomicAdd(out + out_index, scale * x[in_index]);
    else
        atomicAdd(x + in_index, scale * out[out_index]);
}

int main() {
    // Example usage
    size_t N = 1000;
    int w = 16, h = 16, c = 3, batch = 4, stride = 2, forward = 1;
    float scale = 0.5;  // Set your values accordingly
    float *x, *out;  // Assuming these arrays are allocated and initialized

    // Set the CUDA device
    cudaSetDevice(0);

    // Allocate device memory
    float *d_x, *d_out;
    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_out, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Configure the CUDA kernel launch
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);

    // Launch the CUDA kernel
    upsample_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, d_x, w, h, c, batch, stride, forward, scale, d_out);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_out);

    return 0;
}
