#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void clearLabel(float* prA, float* prB, unsigned int num_nodes, float base) {
    unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < num_nodes) {
        prA[id] = base + prA[id] * 0.85;
        prB[id] = 0;
    }
}

int main() {
    // 定义数组大小
    const unsigned int num_nodes = 1000;

    // 分配主机端内存
    float* h_prA = (float*)malloc(num_nodes * sizeof(float));
    float* h_prB = (float*)malloc(num_nodes * sizeof(float));

    // 初始化数组数据
    for (unsigned int i = 0; i < num_nodes; ++i) {
        h_prA[i] = static_cast<float>(i);
        h_prB[i] = static_cast<float>(2 * i);
    }

    // 分配设备端内存
    float* d_prA;
    float* d_prB;
    cudaMalloc((void**)&d_prA, num_nodes * sizeof(float));
    cudaMalloc((void**)&d_prB, num_nodes * sizeof(float));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_prA, h_prA, num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prB, h_prB, num_nodes * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((num_nodes + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    clearLabel<<<gridSize, blockSize>>>(d_prA, d_prB, num_nodes, 1.0);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_prA, d_prA, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prB, d_prB, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (unsigned int i = 0; i < 10; ++i) {
        printf("prA[%u]: %f, prB[%u]: %f\n", i, h_prA[i], i, h_prB[i]);
    }

    // 释放内存
    free(h_prA);
    free(h_prB);
    cudaFree(d_prA);
    cudaFree(d_prB);

    return 0;
}
