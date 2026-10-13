#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void upsweep_scan(int twod, int N, int* output) {
    int twod1 = twod * 2;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * twod1;

    if (idx + twod1 - 1 < N) {
        output[idx + twod1 - 1] += output[idx + twod - 1];
    }
}

int main() {
    // 定义数组大小和 twod 的值
    const int N = 1000;
    const int twod = 32;  // 你的实际需求中使用正确的值

    // 分配主机端内存
    int* h_output = (int*)malloc(N * sizeof(int));

    // 初始化数组数据
    for (int i = 0; i < N; ++i) {
        h_output[i] = i * 2;  // Just an example, you can initialize it according to your needs
    }

    // 分配设备端内存
    int* d_output;
    cudaMalloc((void**)&d_output, N * sizeof(int));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_output, h_output, N * sizeof(int), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    upsweep_scan<<<gridSize, blockSize>>>(twod, N, d_output);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("h_output[%d]: %d\n", i, h_output[i]);
    }

    // 释放内存
    free(h_output);
    cudaFree(d_output);

    return 0;
}
