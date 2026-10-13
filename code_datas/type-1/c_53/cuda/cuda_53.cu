#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cmath>

__global__ void pow_kernel(int N, float ALPHA, float* X, int INCX, float* Y, int INCY) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) {
        Y[i * INCY] = pow(X[i * INCX], ALPHA);
    }
}

int main() {
    // 定义数组大小
    const int N = 1000;

    // 分配主机端内存
    float* h_X = (float*)malloc(N * sizeof(float));
    float* h_Y = (float*)malloc(N * sizeof(float));

    // 初始化数组数据
    for (int i = 0; i < N; ++i) {
        h_X[i] = static_cast<float>(i);
    }

    // 分配设备端内存
    float* d_X;
    float* d_Y;
    cudaMalloc((void**)&d_X, N * sizeof(float));
    cudaMalloc((void**)&d_Y, N * sizeof(float));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    pow_kernel<<<gridSize, blockSize>>>(N, 2.0, d_X, 1, d_Y, 1);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("Y[%d]: %f\n", i, h_Y[i]);
    }

    // 释放内存
    free(h_X);
    free(h_Y);
    cudaFree(d_X);
    cudaFree(d_Y);

    return 0;
}
