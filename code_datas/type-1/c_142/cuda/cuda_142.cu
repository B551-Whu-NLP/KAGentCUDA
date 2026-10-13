#include <stdio.h>

// CUDA kernel
__global__ void getOffsetBox(const int* clsIndex, const float* max_coordinate, float* offset, int dims, int batchSize, const float* before_nms_boxes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= dims) {
        return;
    }

    int numPerbatch = dims;

    for (int i = 0; i < batchSize; i++) {
        if (before_nms_boxes[i * dims * 4 + tid * 4] == (-1)) {
            offset[i * numPerbatch + tid] = 0;
        } else {
            offset[i * numPerbatch + tid] = clsIndex[i * numPerbatch + tid] * (max_coordinate[i * dims * 4] + 1);
        }
    }
}

int main() {
    // Your main program logic here

    // Example: Allocate and initialize host and device memory
    int dims = 256;  // Replace with your actual dimension
    int batchSize = 128;  // Replace with your actual batch size

    int* h_clsIndex = /* Your initialization */;
    float* h_max_coordinate = /* Your initialization */;
    float* h_offset = (float*)malloc(batchSize * dims * sizeof(float));
    float* h_before_nms_boxes = /* Your initialization */;

    int* d_clsIndex;
    float* d_max_coordinate, *d_offset, *d_before_nms_boxes;
    cudaMalloc((void**)&d_clsIndex, batchSize * dims * sizeof(int));
    cudaMalloc((void**)&d_max_coordinate, batchSize * dims * 4 * sizeof(float));
    cudaMalloc((void**)&d_offset, batchSize * dims * sizeof(float));
    cudaMalloc((void**)&d_before_nms_boxes, batchSize * dims * 4 * sizeof(float));

    // Copy host memory to device
    cudaMemcpy(d_clsIndex, h_clsIndex, batchSize * dims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_coordinate, h_max_coordinate, batchSize * dims * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_before_nms_boxes, h_before_nms_boxes, batchSize * dims * 4 * sizeof(float), cudaMemcpyHostToDevice);

    // Configure and launch the CUDA kernel
    dim3 blockSize(256); // Adjust block dimensions based on your requirements
    dim3 gridSize((dims + blockSize.x - 1) / blockSize.x, 1);

    getOffsetBox<<<gridSize, blockSize>>>(d_clsIndex, d_max_coordinate, d_offset, dims, batchSize, d_before_nms_boxes);

    // Copy device memory back to host
    cudaMemcpy(h_offset, d_offset, batchSize * dims * sizeof(float), cudaMemcpyDeviceToHost);

    // Your post-kernel logic here

    // Free allocated memory
    free(h_offset);
    cudaFree(d_clsIndex);
    cudaFree(d_max_coordinate);
    cudaFree(d_offset);
    cudaFree(d_before_nms_boxes);

    return 0;
}
