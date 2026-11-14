#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cmath>

__global__ void kComputeActs(const float* d_nets, float* d_acts) {
    int un_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float tact = 1.0f / (1.0f + expf(-d_acts[un_idx]));
    __syncthreads();
    d_acts[un_idx] = tact;
}

int main() {
    // 定义数组大小
    const int N = 1000;

    // 分配主机端内存
    float* h_nets = (float*)malloc(N * sizeof(float));
    float* h_acts = (float*)malloc(N * sizeof(float));

    // 初始化数组数据
    for (int i = 0; i < N; ++i) {
        h_nets[i] = static_cast<float>(i);
        h_acts[i] = static_cast<float>(i);
    }

    // 分配设备端内存
    float* d_nets;
    float* d_acts;
    cudaMalloc((void**)&d_nets, N * sizeof(float));
    cudaMalloc((void**)&d_acts, N * sizeof(float));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_nets, h_nets, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_acts, h_acts, N * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    kComputeActs<<<gridSize, blockSize>>>(d_nets, d_acts);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_acts, d_acts, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("acts[%d]: %f\n", i, h_acts[i]);
    }

    // 释放内存
    free(h_nets);
    free(h_acts);
    cudaFree(d_nets);
    cudaFree(d_acts);

    return 0;
}
