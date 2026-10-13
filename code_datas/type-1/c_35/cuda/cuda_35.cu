#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void activate_array_leaky_kernel(float* x, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        float val = x[index];
        x[index] = (val > 0) ? val : val / 10;
    }
}

int main() {
    // 定义数组大小
    const int arraySize = 1000;

    // 分配主机端内存
    float* h_x = (float*)malloc(arraySize * sizeof(float));

    // 初始化数组数据
    for (int i = 0; i < arraySize; ++i) {
        h_x[i] = static_cast<float>(i - 500);  // 为了产生正数和负数
    }

    // 分配设备端内存
    float* d_x;
    cudaMalloc((void**)&d_x, arraySize * sizeof(float));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_x, h_x, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    activate_array_leaky_kernel<<<gridSize, blockSize>>>(d_x, arraySize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_x, d_x, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_x[i]);
    }

    // 释放内存
    free(h_x);
    cudaFree(d_x);

    return 0;
}
