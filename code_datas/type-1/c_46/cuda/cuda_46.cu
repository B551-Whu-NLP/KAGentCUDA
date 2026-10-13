#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void doubleArrayVectorAddKernel(double* d_in_a, double* d_in_b, double* d_out, int length) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < length) {
        d_out[tid] = d_in_a[tid] + d_in_b[tid];
    }
}

int main() {
    // 定义数组大小
    const int arraySize = 1000;

    // 分配主机端内存
    double* h_d_in_a = (double*)malloc(arraySize * sizeof(double));
    double* h_d_in_b = (double*)malloc(arraySize * sizeof(double));
    double* h_d_out = (double*)malloc(arraySize * sizeof(double));

    // 初始化数组数据
    for (int i = 0; i < arraySize; ++i) {
        h_d_in_a[i] = static_cast<double>(i);
        h_d_in_b[i] = static_cast<double>(2 * i);
    }

    // 分配设备端内存
    double* d_d_in_a;
    double* d_d_in_b;
    double* d_d_out;
    cudaMalloc((void**)&d_d_in_a, arraySize * sizeof(double));
    cudaMalloc((void**)&d_d_in_b, arraySize * sizeof(double));
    cudaMalloc((void**)&d_d_out, arraySize * sizeof(double));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_d_in_a, h_d_in_a, arraySize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_in_b, h_d_in_b, arraySize * sizeof(double), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((arraySize + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    doubleArrayVectorAddKernel<<<gridSize, blockSize>>>(d_d_in_a, d_d_in_b, d_d_out, arraySize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_d_out, d_d_out, arraySize * sizeof(double), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_d_out[i]);
    }

    // 释放内存
    free(h_d_in_a);
    free(h_d_in_b);
    free(h_d_out);
    cudaFree(d_d_in_a);
    cudaFree(d_d_in_b);
    cudaFree(d_d_out);

    return 0;
}
