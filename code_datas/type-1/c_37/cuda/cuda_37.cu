#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void add_kernel(float* inputleft, float* inputright, float* output, int count) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < count) {
        output[idx] = inputleft[idx] + inputright[idx];
    }
}

int main() {
    // 定义数组大小
    const int arraySize = 1000;

    // 分配主机端内存
    float* h_inputleft = (float*)malloc(arraySize * sizeof(float));
    float* h_inputright = (float*)malloc(arraySize * sizeof(float));
    float* h_output = (float*)malloc(arraySize * sizeof(float));

    // 初始化数组数据
    for (int i = 0; i < arraySize; ++i) {
        h_inputleft[i] = static_cast<float>(i);
        h_inputright[i] = static_cast<float>(2 * i);
    }

    // 分配设备端内存
    float* d_inputleft;
    float* d_inputright;
    float* d_output;
    cudaMalloc((void**)&d_inputleft, arraySize * sizeof(float));
    cudaMalloc((void**)&d_inputright, arraySize * sizeof(float));
    cudaMalloc((void**)&d_output, arraySize * sizeof(float));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_inputleft, h_inputleft, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputright, h_inputright, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    add_kernel<<<gridSize, blockSize>>>(d_inputleft, d_inputright, d_output, arraySize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_output, d_output, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_output[i]);
    }

    // 释放内存
    free(h_inputleft);
    free(h_inputright);
    free(h_output);
    cudaFree(d_inputleft);
    cudaFree(d_inputright);
    cudaFree(d_output);

    return 0;
}
