#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel_rows(const float *filter, const float *input, float *output, int imageW, int imageH, int filterR) {
    int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_y = threadIdx.y + blockDim.y * blockIdx.y;
    int grid_width = gridDim.x * blockDim.x;
    int idx = grid_width * idx_y + idx_x;
    float sum = 0;

    for (int k = -filterR; k <= filterR; k++) {
        int d = idx_x + k;
        if (d >= 0 && d < imageW) {
            sum += input[idx_y * imageW + d] * filter[filterR - k];
        }
    }

    output[idx] = sum;
}

int main() {
    // Set image dimensions and filter radius
    int imageW = 128;  // Set the appropriate value
    int imageH = 128;  // Set the appropriate value
    int filterR = 3;   // Set the appropriate value

    // Allocate host memory
    float *h_filter, *h_input, *h_output;
    h_filter = (float *)malloc((2 * filterR + 1) * sizeof(float));
    h_input = (float *)malloc(imageW * imageH * sizeof(float));
    h_output = (float *)malloc(imageW * imageH * sizeof(float));

    // Initialize filter and input arrays (you may use your own initialization logic)
    for (int i = 0; i < 2 * filterR + 1; i++) {
        h_filter[i] = 1.0f / (2 * filterR + 1); // Example: Box filter
    }

    for (int i = 0; i < imageW * imageH; i++) {
        h_input[i] = static_cast<float>(i % 255); // Example: Simple input
    }

    // Allocate device memory
    float *d_filter, *d_input, *d_output;
    cudaMalloc((void **)&d_filter, (2 * filterR + 1) * sizeof(float));
    cudaMalloc((void **)&d_input, imageW * imageH * sizeof(float));
    cudaMalloc((void **)&d_output, imageW * imageH * sizeof(float));

    // Copy arrays from host to device
    cudaMemcpy(d_filter, h_filter, (2 * filterR + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 blockSize(16, 16); // You may adjust the block size
    dim3 gridSize((imageW + blockSize.x - 1) / blockSize.x, (imageH + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    kernel_rows<<<gridSize, blockSize>>>(d_filter, d_input, d_output, imageW, imageH, filterR);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Check for errors during the kernel launch
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    // Copy the result array from device to host
    cudaMemcpy(h_output, d_output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result array (you may modify this part based on your needs)
    printf("Output Array:\n");
    for (int i = 0; i < imageH; i++) {
        for (int j = 0; j < imageW; j++) {
            printf("%.2f ", h_output[i * imageW + j]);
        }
        printf("\n");
    }

    // Cleanup
    free(h_filter);
    free(h_input);
    free(h_output);
    cudaFree(d_filter);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
 
