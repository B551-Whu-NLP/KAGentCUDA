#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void set_valid_mask(const float* score, float score_thr, int* valid_mask, int dims) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < dims) {
        valid_mask[tid] = (score[tid] > score_thr) ? 1 : 0;
    }
}

int main() {
    // 定义数组大小
    const int dims = 1024;

    // 分配主机端内存
    float* h_score = (float*)malloc(dims * sizeof(float));
    int* h_valid_mask = (int*)malloc(dims * sizeof(int));

    // 初始化数组数据
    for (int i = 0; i < dims; ++i) {
        h_score[i] = static_cast<float>(i); // Just an example, you can initialize it according to your needs
    }

    // 分配设备端内存
    float* d_score;
    int* d_valid_mask;
    cudaMalloc((void**)&d_score, dims * sizeof(float));
    cudaMalloc((void**)&d_valid_mask, dims * sizeof(int));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_score, h_score, dims * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((dims + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    set_valid_mask<<<gridSize, blockSize>>>(d_score, 500.0f, d_valid_mask, dims);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_valid_mask, d_valid_mask, dims * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("h_valid_mask[%d]: %d\n", i, h_valid_mask[i]);
    }

    // 释放内存
    free(h_score);
    free(h_valid_mask);
    cudaFree(d_score);
    cudaFree(d_valid_mask);

    return 0;
}
