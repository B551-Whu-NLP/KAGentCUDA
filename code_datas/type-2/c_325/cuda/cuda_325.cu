#include <stdio.h>
#include <cuda_runtime.h>

// Kernel function
__global__ void GPU_array_rowKernel(double *input, double *output, int length) {
    int xCuda = blockDim.x * blockIdx.x + threadIdx.x;
    int yCuda = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = yCuda * length + xCuda;

    if (xCuda >= length || yCuda >= length)
        return;

    if (xCuda == 0 || xCuda == length - 1) {
        output[idx] = 0;
        return;
    }

    output[idx] = input[idx];
    output[idx] += xCuda == 0 ? 0 : input[idx - 1];
    output[idx] += xCuda == length - 1 ? 0 : input[idx + 1];
}

int main() {
    // Set array length
    int length = 10;  // Set the appropriate value

    // Allocate host memory
    double *h_input, *h_output;
    h_input = (double *)malloc(length * length * sizeof(double));
    h_output = (double *)malloc(length * length * sizeof(double));

    // Initialize input array (you may use your own initialization logic)
    for (int i = 0; i < length * length; i++) {
        h_input[i] = i;  // Example: Simple initialization
    }

    // Allocate device memory
    double *d_input, *d_output;
    cudaMalloc((void **)&d_input, length * length * sizeof(double));
    cudaMalloc((void **)&d_output, length * length * sizeof(double));

    // Copy input array from host to device
    cudaMemcpy(d_input, h_input, length * length * sizeof(double), cudaMemcpyHostToDevice);

    // Set grid and block sizes
    dim3 blockSize(16, 16);  // You may adjust the block size
    dim3 gridSize((length + blockSize.x - 1) / blockSize.x, (length + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    GPU_array_rowKernel<<<gridSize, blockSize>>>(d_input, d_output, length);

    // Synchronize to make sure the kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Check for errors during the kernel launch
    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    // Copy the result array from device to host
    cudaMemcpy(h_output, d_output, length * length * sizeof(double), cudaMemcpyDeviceToHost);

    // Display the result array (you may modify this part based on your needs)
    printf("Output Array:\n");
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < length; j++) {
            printf("%f ", h_output[i * length + j]);
        }
        printf("\n");
    }

    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
 
