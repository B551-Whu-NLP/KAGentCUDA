#include <stdio.h>

// CUDA kernel
__global__ void kernel_columns(const float* filter, const float* buffer, float* output, int imageW, int imageH, int filterR) {
    int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_y = threadIdx.y + blockDim.y * blockIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int idx = grid_width * idx_y + idx_x;

    float sum = 0;

    for (int k = -filterR; k <= filterR; k++) {
        int d = idx_y + k;

        if (d >= 0 && d < imageH) {
            sum += buffer[d * imageW + idx_x] * filter[filterR - k];
        }
    }

    output[idx] = sum;
}

int main() {
    // Your main program logic here

    // Example: Allocate and initialize host and device memory
    int imageW = 512;    // Replace with your actual width
    int imageH = 512;    // Replace with your actual height
    int filterR = 3;     // Replace with your actual filter radius

    float* h_filter = (float*)malloc((2 * filterR + 1) * sizeof(float));
    float* h_buffer = (float*)malloc(imageW * imageH * sizeof(float));
    float* h_output = (float*)malloc(imageW * imageH * sizeof(float));

    float* d_filter, * d_buffer, * d_output;
    cudaMalloc((void**)&d_filter, (2 * filterR + 1) * sizeof(float));
    cudaMalloc((void**)&d_buffer, imageW * imageH * sizeof(float));
    cudaMalloc((void**)&d_output, imageW * imageH * sizeof(float));

    // Copy host memory to device
    cudaMemcpy(d_filter, h_filter, (2 * filterR + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buffer, h_buffer, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice);

    // Configure and launch the CUDA kernel
    dim3 blockSize(16, 16); // Adjust block dimensions based on your requirements
    dim3 gridSize((imageW + blockSize.x - 1) / blockSize.x, (imageH + blockSize.y - 1) / blockSize.y);

    kernel_columns<<<gridSize, blockSize>>>(d_filter, d_buffer, d_output, imageW, imageH, filterR);

    // Copy device memory back to host
    cudaMemcpy(h_output, d_output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost);

    // Your post-kernel logic here

    // Free allocated memory
    free(h_filter);
    free(h_buffer);
    free(h_output);
    cudaFree(d_filter);
    cudaFree(d_buffer);
    cudaFree(d_output);

    return 0;
}
