#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void boundaryCorrectIndexesKernel(int* d_in, int* d_out, int length, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        if (d_in[tid] > N) {
            d_out[tid] = N;
        } else {
            d_out[tid] = d_in[tid];
        }
    }
}

int main() {
    // 定义数组大小和 N 的值
    const int length = 1000;
    const int N = 500;

    // 分配主机端内存
    int* h_in = (int*)malloc(length * sizeof(int));
    int* h_out = (int*)malloc(length * sizeof(int));

    // 初始化数组数据
    for (int i = 0; i < length; ++i) {
        h_in[i] = i * 2; // Just an example, you can initialize it according to your needs
    }

    // 分配设备端内存
    int* d_in;
    int* d_out;
    cudaMalloc((void**)&d_in, length * sizeof(int));
    cudaMalloc((void**)&d_out, length * sizeof(int));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_in, h_in, length * sizeof(int), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((length + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    boundaryCorrectIndexesKernel<<<gridSize, blockSize>>>(d_in, d_out, length, N);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_out, d_out, length * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("h_out[%d]: %d\n", i, h_out[i]);
    }

    // 释放内存
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
