#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void testInt1(const int* input, int dims) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= dims) {
        return;
    }

    int sum = 0;

    for (int i = 0; i < 3000 * 4; i++) {
        if (input[i] == 0) {
            sum++;
        }
    }
}

int main() {
    // 定义数组大小
    const int dims = 3000 * 4;

    // 分配主机端内存
    int* h_input = (int*)malloc(dims * sizeof(int));

    // 初始化数组数据
    for (int i = 0; i < dims; ++i) {
        h_input[i] = i % 2; // Filling with alternating 0 and 1 for testing
    }

    // 分配设备端内存
    int* d_input;
    cudaMalloc((void**)&d_input, dims * sizeof(int));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_input, h_input, dims * sizeof(int), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((dims + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    testInt1<<<gridSize, blockSize>>>(d_input, dims);

    // 等待 GPU 完成所有任务
    cudaDeviceSynchronize();

    // 释放内存
    free(h_input);
    cudaFree(d_input);

    return 0;
}
