#include <stdio.h>

// Define the CUDA kernel
__global__ void shortcut_kernel(int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1,
                                float *add, int w2, int h2, int c2, float s1, float s2, float *out) {
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (id >= size)
        return;

    int i = id % minw;
    id /= minw;
    int j = id % minh;
    id /= minh;
    int k = id % minc;
    id /= minc;
    int b = id % batch;

    int out_index = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
    int add_index = i * stride + w1 * (j * stride + h1 * (k + c1 * b));

    out[out_index] = s1 * out[out_index] + s2 * add[add_index];
}

int main() {
    // Example usage
    int size = 1000;  // Set your value of size accordingly
    int minw = 16, minh = 16, minc = 3, stride = 2, sample = 2, batch = 4;
    int w1 = 8, h1 = 8, c1 = 3, w2 = 8, h2 = 8, c2 = 3;
    float s1 = 0.5, s2 = 0.5;  // Set your values accordingly
    float *add, *out;          // Assuming these arrays are allocated and initialized

    // Set the CUDA device
    cudaSetDevice(0);

    // Allocate device memory
    float *d_add, *d_out;
    cudaMalloc((void **)&d_add, minw * stride * sizeof(float));
    cudaMalloc((void **)&d_out, size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_add, add, minw * stride * sizeof(float), cudaMemcpyHostToDevice);

    // Configure the CUDA kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    shortcut_kernel<<<blocksPerGrid, threadsPerBlock>>>(size, minw, minh, minc, stride, sample, batch, w1, h1, c1, d_add,
                                                        w2, h2, c2, s1, s2, d_out);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_add);
    cudaFree(d_out);

    return 0;
}
