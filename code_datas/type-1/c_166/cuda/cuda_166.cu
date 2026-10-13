#include <stdio.h>

// Define the CUDA kernel
__global__ void get_before_nms_data(const float *boxes, const float *scores, const int *labels, const int *index,
                                     float *boxes_out, float *scores_out, int *labels_out, int dims) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= dims) {
        return;
    }

    if (index[tid] == 0) {
        boxes_out[tid * 4 + 0] = -1;
        boxes_out[tid * 4 + 1] = -1;
        boxes_out[tid * 4 + 2] = -1;
        boxes_out[tid * 4 + 3] = -1;
        scores_out[tid] = -1;
        labels_out[tid] = -1;
    } else {
        boxes_out[tid * 4 + 0] = boxes[tid * 4 + 0];
        boxes_out[tid * 4 + 1] = boxes[tid * 4 + 1];
        boxes_out[tid * 4 + 2] = boxes[tid * 4 + 2];
        boxes_out[tid * 4 + 3] = boxes[tid * 4 + 3];
        scores_out[tid] = scores[tid];
        labels_out[tid] = labels[tid];
    }
}

int main() {
    // Example usage
    int dims = 1000;  // Set your value of dims accordingly
    float *boxes, *scores, *boxes_out, *scores_out;  // Assuming these arrays are allocated and initialized
    int *labels, *index, *labels_out;                // Assuming these arrays are allocated and initialized

    // Set the CUDA device
    cudaSetDevice(0);

    // Allocate device memory
    float *d_boxes, *d_scores, *d_boxes_out, *d_scores_out;
    int *d_labels, *d_index, *d_labels_out;

    cudaMalloc((void **)&d_boxes, dims * 4 * sizeof(float));
    cudaMalloc((void **)&d_scores, dims * sizeof(float));
    cudaMalloc((void **)&d_labels, dims * sizeof(int));
    cudaMalloc((void **)&d_index, dims * sizeof(int));
    cudaMalloc((void **)&d_boxes_out, dims * 4 * sizeof(float));
    cudaMalloc((void **)&d_scores_out, dims * sizeof(float));
    cudaMalloc((void **)&d_labels_out, dims * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_boxes, boxes, dims * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scores, scores, dims * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, dims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index, index, dims * sizeof(int), cudaMemcpyHostToDevice);

    // Configure the CUDA kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (dims + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    get_before_nms_data<<<blocksPerGrid, threadsPerBlock>>>(d_boxes, d_scores, d_labels, d_index,
                                                            d_boxes_out, d_scores_out, d_labels_out, dims);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(boxes_out, d_boxes_out, dims * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(scores_out, d_scores_out, dims * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(labels_out, d_labels_out, dims * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaFree(d_labels);
    cudaFree(d_index);
    cudaFree(d_boxes_out);
    cudaFree(d_scores_out);
    cudaFree(d_labels_out);

    return 0;
}
