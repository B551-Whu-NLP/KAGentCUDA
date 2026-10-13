#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cmath>

__global__ void squareKernel(float* d_in, float* d_out, int N) {
    const unsigned int lid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + lid;
    if (gid < N) {
        d_out[gid] = pow(d_in[gid] / (d_in[gid] - 2.3), 3);
    }
}

int main() {
    // 定义数组大小
    const int arraySize = 1000;

    // 分配主机端内存
    float* h_d_in = (float*)malloc(arraySize * sizeof(float));
    float* h_d_out = (float*)malloc(arraySize * sizeof(float));

    // 初始化数组数据
    for (int i = 0; i < arraySize; ++i) {
        h_d_in[i] = static_cast<float>(i);
    }

    // 分配设备端内存
    float* d_d_in;
    float* d_d_out;
    cudaMalloc((void**)&d_d_in, arraySize * sizeof(float));
    cudaMalloc((void**)&d_d_out, arraySize * sizeof(float));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_d_in, h_d_in, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((arraySize + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    squareKernel<<<gridSize, blockSize>>>(d_d_in, d_d_out, arraySize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_d_out, d_d_out, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_d_out[i]);
    }

    // 释放内存
    free(h_d_in);
    free(h_d_out);
    cudaFree(d_d_in);
    cudaFree(d_d_out);

    return 0;
}
