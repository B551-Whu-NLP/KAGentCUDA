#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void mult_add_into_kernel(int n, float* a, float* b, float* c) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] += a[i] * b[i];
    }
}

int main() {
    // 定义数组大小
    const int arraySize = 1000;

    // 分配主机端内存
    float* h_a = (float*)malloc(arraySize * sizeof(float));
    float* h_b = (float*)malloc(arraySize * sizeof(float));
    float* h_c = (float*)malloc(arraySize * sizeof(float));

    // 初始化数组数据
    for (int i = 0; i < arraySize; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
        h_c[i] = static_cast<float>(3 * i);
    }

    // 分配设备端内存
    float* d_a;
    float* d_b;
    float* d_c;
    cudaMalloc((void**)&d_a, arraySize * sizeof(float));
    cudaMalloc((void**)&d_b, arraySize * sizeof(float));
    cudaMalloc((void**)&d_c, arraySize * sizeof(float));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_a, h_a, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((arraySize + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    mult_add_into_kernel<<<gridSize, blockSize>>>(arraySize, d_a, d_b, d_c);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_c, d_c, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_c[i]);
    }

    // 释放内存
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
