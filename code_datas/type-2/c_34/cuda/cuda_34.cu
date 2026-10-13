#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// 定义矩阵的维度
#define N 16

__global__ void addKernel(int* c, const int* a, const int* b) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = y * blockDim.x + x;
    c[i] = a[i] + b[i];
}

int main() {
    // 定义矩阵
    int a[N][N], b[N][N], c[N][N];

    // 分配设备端内存
    int* d_a;
    int* d_b;
    int* d_c;
    cudaMalloc((void**)&d_a, N * N * sizeof(int));
    cudaMalloc((void**)&d_b, N * N * sizeof(int));
    cudaMalloc((void**)&d_c, N * N * sizeof(int));

    // 初始化矩阵数据
    for (int i = 0; i < N * N; ++i) {
        a[i / N][i % N] = i;
        b[i / N][i % N] = 2 * i;
    }

    // 将矩阵数据从主机端拷贝到设备端
    cudaMemcpy(d_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(N, N);
    dim3 gridSize(1, 1);

    // 启动内核
    addKernel<<<gridSize, blockSize>>>(d_c, d_a, d_b);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    // 释放内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
