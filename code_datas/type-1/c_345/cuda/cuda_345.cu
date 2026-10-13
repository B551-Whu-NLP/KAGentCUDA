#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

__global__ void returnResult(const float *box, const float *score, const int *label, float *box_out, float *score_out, int *label_out, float score_thr, const int dims) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= dims) {
        return;
    }

    if (score[tid] < score_thr) {
        score_out[tid] = 0;
        box_out[tid * 4 + 0] = -1;
        box_out[tid * 4 + 1] = -1;
        box_out[tid * 4 + 2] = -1;
        box_out[tid * 4 + 3] = -1;
        label_out[tid] = -1;
    } else {
        score_out[tid] = score[tid];
        box_out[tid * 4 + 0] = box[tid * 4 + 0];
        box_out[tid * 4 + 1] = box[tid * 4 + 1];
        box_out[tid * 4 + 2] = box[tid * 4 + 2];
        box_out[tid * 4 + 3] = box[tid * 4 + 3];
        label_out[tid] = label[tid];
    }
}

int main() {
    const int dims = 1024; // Adjust the size based on your data
    const float score_thr = 0.5; // Adjust the threshold based on your requirement

    float *box, *score_out, *box_out;
    int *label, *label_out;

    size_t size = dims * sizeof(float);

    // Allocate host memory
    box = (float *)malloc(size * 4);
    score_out = (float *)malloc(size);
    box_out = (float *)malloc(size * 4);
    label = (int *)malloc(size);
    label_out = (int *)malloc(size);

    // Initialize host data (you may need to modify this based on your use case)
    for (int i = 0; i < dims * 4; ++i) {
        box[i] = static_cast<float>(i);
    }

    for (int i = 0; i < dims; ++i) {
        score_out[i] = static_cast<float>(i);
        label[i] = i;
    }

    // Allocate device memory
    float *d_box, *d_score_out, *d_box_out;
    int *d_label, *d_label_out;
    cudaMalloc((void **)&d_box, size * 4);
    cudaMalloc((void **)&d_score_out, size);
    cudaMalloc((void **)&d_box_out, size * 4);
    cudaMalloc((void **)&d_label, size);
    cudaMalloc((void **)&d_label_out, size);

    // Copy data from host to device
    cudaMemcpy(d_box, box, size * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_score_out, score_out, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_label, label, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((dims + block_size.x - 1) / block_size.x);

    // Launch the kernel
    returnResult<<<grid_size, block_size>>>(d_box, d_score_out, d_label, d_box_out, d_score_out, d_label_out, score_thr, dims);

    // Copy data from device to host
    cudaMemcpy(score_out, d_score_out, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(box_out, d_box_out, size * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(label_out, d_label_out, size, cudaMemcpyDeviceToHost);

    // Free device and host memory
    cudaFree(d_box);
    cudaFree(d_score_out);
    cudaFree(d_box_out);
    cudaFree(d_label);
    cudaFree(d_label_out);
    free(box);
    free(score_out);
    free(box_out);
    free(label);
    free(label_out);

    return 0;
}
 
