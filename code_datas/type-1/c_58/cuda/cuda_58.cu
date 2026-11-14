#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void incKernel(int* g_out, const int* g_in, int N, int inner_reps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        for (int i = 0; i < inner_reps; ++i) {
            g_out[idx] = g_in[idx] + 1;
        }
    }
}

int main() {
    // 定义数组大小和重复次数
    const int N = 1000;
    const int inner_reps = 1000;

    // 分配主机端内存
    int* h_in = (int*)malloc(N * sizeof(int));
    int* h_out = (int*)malloc(N * sizeof(int));

    // 初始化数组数据
    for (int i = 0; i < N; ++i) {
        h_in[i] = i;
    }

    // 分配设备端内存
    int* d_in;
    int* d_out;
    cudaMalloc((void**)&d_in, N * sizeof(int));
    cudaMalloc((void**)&d_out, N * sizeof(int));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    incKernel<<<gridSize, blockSize>>>(d_out, d_in, N, inner_reps);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

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
