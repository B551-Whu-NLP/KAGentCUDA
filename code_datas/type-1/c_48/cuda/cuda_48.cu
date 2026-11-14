#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void evenoddincrement(float* g_data, int even_inc, int odd_inc) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    if ((tx % 2) == 0) {
        g_data[tx] += even_inc;
    } else {
        g_data[tx] += odd_inc;
    }
}

int main() {
    // 定义数组大小
    const int arraySize = 1000;

    // 分配主机端内存
    float* h_data = (float*)malloc(arraySize * sizeof(float));

    // 初始化数组数据
    for (int i = 0; i < arraySize; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // 分配设备端内存
    float* d_data;
    cudaMalloc((void**)&d_data, arraySize * sizeof(float));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_data, h_data, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((arraySize + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    evenoddincrement<<<gridSize, blockSize>>>(d_data, 2, 3);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_data, d_data, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_data[i]);
    }

    // 释放内存
    free(h_data);
    cudaFree(d_data);

    return 0;
}
