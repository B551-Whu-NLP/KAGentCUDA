#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void operacionKernelGPU(float* u, float* lu, float u_m, float u_d, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        lu[idx] = (u[idx] - u_m) / u_d;
    }
}

int main() {
    // 定义数组大小
    const int arraySize = 1000;

    // 分配主机端内存
    float* h_u = (float*)malloc(arraySize * sizeof(float));
    float* h_lu = (float*)malloc(arraySize * sizeof(float));

    // 初始化数组数据
    for (int i = 0; i < arraySize; ++i) {
        h_u[i] = static_cast<float>(i);
    }

    // 分配设备端内存
    float* d_u;
    float* d_lu;
    cudaMalloc((void**)&d_u, arraySize * sizeof(float));
    cudaMalloc((void**)&d_lu, arraySize * sizeof(float));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_u, h_u, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((arraySize + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    operacionKernelGPU<<<gridSize, blockSize>>>(d_u, d_lu, 5.0f, 2.0f, arraySize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_lu, d_lu, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_lu[i]);
    }

    // 释放内存
    free(h_u);
    free(h_lu);
    cudaFree(d_u);
    cudaFree(d_lu);

    return 0;
}
