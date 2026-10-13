#include <stdio.h>

// CUDA kernel
__global__ void forward_avgpool_layer_kernel(int n, int w, int h, int c, float* input, float* output) {
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (id >= n)
        return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c * b);
    output[out_index] = 0;

    for (i = 0; i < w * h; ++i) {
        int in_index = i + h * w * (k + b * c);
        output[out_index] += input[in_index];
    }

    output[out_index] /= w * h;
}

int main() {
    // Your main program logic here

    // Example: Allocate and initialize host and device memory
    int n = 100; // Replace with your actual size
    int w = 32;  // Replace with your actual width
    int h = 32;  // Replace with your actual height
    int c = 3;   // Replace with your actual channels

    float* h_input = (float*)malloc(n * w * h * c * sizeof(float));
    float* h_output = (float*)malloc(n * c * sizeof(float));

    float* d_input, * d_output;
    cudaMalloc((void**)&d_input, n * w * h * c * sizeof(float));
    cudaMalloc((void**)&d_output, n * c * sizeof(float));

    // Copy host memory to device
    cudaMemcpy(d_input, h_input, n * w * h * c * sizeof(float), cudaMemcpyHostToDevice);

    // Configure and launch the CUDA kernel
    dim3 blockSize(256); // Adjust block dimensions based on your requirements
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, 1); // 1D grid

    forward_avgpool_layer_kernel<<<gridSize, blockSize>>>(n, w, h, c, d_input, d_output);

    // Copy device memory back to host
    cudaMemcpy(h_output, d_output, n * c * sizeof(float), cudaMemcpyDeviceToHost);

    // Your post-kernel logic here

    // Free allocated memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
