#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void saxpy_gpu(float* vecY, float* vecX, float alpha, int n) {
    int x, y, i;
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    i = y * 1024 + x;

    if (i < n) {
        vecY[i] = alpha * vecX[i] + vecY[i];
    }
}

int main() {
    // 定义数组大小
    const int n = 1024 * 1024;

    // 分配主机端内存
    float* h_vecY = (float*)malloc(n * sizeof(float));
    float* h_vecX = (float*)malloc(n * sizeof(float));

    // 初始化数组数据
    for (int i = 0; i < n; ++i) {
        h_vecY[i] = static_cast<float>(i); // Just an example, you can initialize it according to your needs
        h_vecX[i] = static_cast<float>(i * 2); // Just an example, you can initialize it according to your needs
    }

    // 分配设备端内存
    float* d_vecY;
    float* d_vecX;
    cudaMalloc((void**)&d_vecY, n * sizeof(float));
    cudaMalloc((void**)&d_vecX, n * sizeof(float));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_vecY, h_vecY, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vecX, h_vecX, n * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(32, 32); // Assuming 1024 threads in total
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    // 启动内核
    saxpy_gpu<<<gridSize, blockSize>>>(d_vecY, d_vecX, 2.0f, n);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_vecY, d_vecY, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("h_vecY[%d]: %f\n", i, h_vecY[i]);
    }

    // 释放内存
    free(h_vecY);
    free(h_vecX);
    cudaFree(d_vecY);
    cudaFree(d_vecX);

    return 0;
}
