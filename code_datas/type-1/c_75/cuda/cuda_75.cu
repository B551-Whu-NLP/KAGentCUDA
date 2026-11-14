#include <stdio.h>

// CUDA kernel to initialize connected components labeling
__global__ void InitCCL(int labelList[], int reference[], int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int id = x + y * width;
    labelList[id] = reference[id] = id;
}

int main() {
    // Set your desired width and height
    int width = 512;
    int height = 512;

    // Calculate grid and block dimensions
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);
    dim3 blockSize(16, 16);

    // Allocate memory on the host
    int* h_labelList = (int*)malloc(width * height * sizeof(int));
    int* h_reference = (int*)malloc(width * height * sizeof(int));

    // Allocate memory on the device
    int* d_labelList, * d_reference;
    cudaMalloc((void**)&d_labelList, width * height * sizeof(int));
    cudaMalloc((void**)&d_reference, width * height * sizeof(int));

    // Launch the CUDA kernel
    InitCCL<<<gridSize, blockSize>>>(d_labelList, d_reference, width, height);

    // Copy the results back to the host
    cudaMemcpy(h_labelList, d_labelList, width * height * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_reference, d_reference, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_labelList);
    cudaFree(d_reference);

    // Free host memory
    free(h_labelList);
    free(h_reference);

    return 0;
}
