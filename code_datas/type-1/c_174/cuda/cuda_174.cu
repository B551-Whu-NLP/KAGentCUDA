#include <stdio.h>

// Define the CUDA kernel
__global__ void get_boxes_for_nms(const float *boxes_before_nms, const float *offset, float *boxes_for_nms, int dims) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= dims) {
        return;
    }

    if (boxes_before_nms[tid * 4 + 0] == (-1) && boxes_before_nms[tid * 4 + 1] == (-1) &&
        boxes_before_nms[tid * 4 + 2] == (-1) && boxes_before_nms[tid * 4 + 3] == (-1)) {
        boxes_for_nms[tid * 4 + 0] = (-1);
        boxes_for_nms[tid * 4 + 1] = (-1);
        boxes_for_nms[tid * 4 + 2] = (-1);
        boxes_for_nms[tid * 4 + 3] = (-1);
    } else {
        boxes_for_nms[tid * 4 + 0] = boxes_before_nms[tid * 4 + 0] + offset[tid];
        boxes_for_nms[tid * 4 + 1] = boxes_before_nms[tid * 4 + 1] + offset[tid];
        boxes_for_nms[tid * 4 + 2] = boxes_before_nms[tid * 4 + 2] + offset[tid];
        boxes_for_nms[tid * 4 + 3] = boxes_before_nms[tid * 4 + 3] + offset[tid];
    }
}

int main() {
    // Example usage
    int dims = 1000; // Set your value of dims accordingly

    float *boxes_before_nms, *offset, *boxes_for_nms;

    // Assuming these arrays are allocated and initialized
    // ...

    // Set the CUDA device
    cudaSetDevice(0);

    // Allocate device memory
    float *d_boxes_before_nms, *d_offset, *d_boxes_for_nms;

    cudaMalloc((void **)&d_boxes_before_nms, dims * 4 * sizeof(float));
    cudaMalloc((void **)&d_offset, dims * sizeof(float));
    cudaMalloc((void **)&d_boxes_for_nms, dims * 4 * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_boxes_before_nms, boxes_before_nms, dims * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offset, offset, dims * sizeof(float), cudaMemcpyHostToDevice);

    // Configure the CUDA kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (dims + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    get_boxes_for_nms<<<blocksPerGrid, threadsPerBlock>>>(d_boxes_before_nms, d_offset, d_boxes_for_nms, dims);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(boxes_for_nms, d_boxes_for_nms, dims * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_boxes_before_nms);
    cudaFree(d_offset);
    cudaFree(d_boxes_for_nms);

    return 0;
}
