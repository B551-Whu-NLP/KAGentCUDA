#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void Kernel_Function_update_sgd(float lr, float* dev_parameter, float* dev_gradient, int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int N = size;
    while (tid < N) {
        dev_parameter[tid] -= lr * dev_gradient[tid];
        tid += gridDim.x * blockDim.x;
    }
}

int main() {
    // 定义数组大小
    const int arraySize = 1000;

    // 分配主机端内存
    float* h_dev_parameter = (float*)malloc(arraySize * sizeof(float));
    float* h_dev_gradient = (float*)malloc(arraySize * sizeof(float));

    // 初始化数组数据
    for (int i = 0; i < arraySize; ++i) {
        h_dev_parameter[i] = static_cast<float>(i);
        h_dev_gradient[i] = static_cast<float>(2 * i);
    }

    // 分配设备端内存
    float* d_dev_parameter;
    float* d_dev_gradient;
    cudaMalloc((void**)&d_dev_parameter, arraySize * sizeof(float));
    cudaMalloc((void**)&d_dev_gradient, arraySize * sizeof(float));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_dev_parameter, h_dev_parameter, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dev_gradient, h_dev_gradient, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((arraySize + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    Kernel_Function_update_sgd<<<gridSize, blockSize>>>(0.01f, d_dev_parameter, d_dev_gradient, arraySize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_dev_parameter, d_dev_parameter, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_dev_parameter[i]);
    }

    // 释放内存
    free(h_dev_parameter);
    free(h_dev_gradient);
    cudaFree(d_dev_parameter);
    cudaFree(d_dev_gradient);

    return 0;
}
