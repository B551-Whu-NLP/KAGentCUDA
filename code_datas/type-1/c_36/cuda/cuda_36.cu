#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void logistic(unsigned int n, float a, float* x, float* z) {
    unsigned int myId = blockDim.x * blockIdx.x + threadIdx.x;
    if (myId < n) {
        z[myId] = a * x[myId] * (1 - x[myId]);
    }
}

int main() {
    // 定义数组大小
    const unsigned int arraySize = 1000;

    // 分配主机端内存
    float* h_x = (float*)malloc(arraySize * sizeof(float));
    float* h_z = (float*)malloc(arraySize * sizeof(float));

    // 初始化数组数据
    for (unsigned int i = 0; i < arraySize; ++i) {
        h_x[i] = static_cast<float>(i) / arraySize;  // 确保在 0 到 1 之间
    }

    // 分配设备端内存
    float* d_x;
    float* d_z;
    cudaMalloc((void**)&d_x, arraySize * sizeof(float));
    cudaMalloc((void**)&d_z, arraySize * sizeof(float));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_x, h_x, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    logistic<<<gridSize, blockSize>>>(arraySize, 2.0f, d_x, d_z);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_z, d_z, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (unsigned int i = 0; i < 10; ++i) {
        printf("%f ", h_z[i]);
    }

    // 释放内存
    free(h_x);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_z);

    return 0;
}
