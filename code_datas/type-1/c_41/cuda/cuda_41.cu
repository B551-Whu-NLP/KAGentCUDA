#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void InitReduction(bool* flags, int voxelCount, int* reduction, int reductionSize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < reductionSize) {
        reduction[tid] = (tid < voxelCount) ? flags[tid] : 0;
    }
}

int main() {
    // 定义数组大小
    const int voxelCount = 1000;
    const int reductionSize = 1024; // 你的reduction数组的大小

    // 分配主机端内存
    bool* h_flags = (bool*)malloc(voxelCount * sizeof(bool));
    int* h_reduction = (int*)malloc(reductionSize * sizeof(int));

    // 初始化数组数据
    for (int i = 0; i < voxelCount; ++i) {
        h_flags[i] = i % 2 == 0; // 举例：偶数位置设为true，奇数位置设为false
    }

    // 分配设备端内存
    bool* d_flags;
    int* d_reduction;
    cudaMalloc((void**)&d_flags, voxelCount * sizeof(bool));
    cudaMalloc((void**)&d_reduction, reductionSize * sizeof(int));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_flags, h_flags, voxelCount * sizeof(bool), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((reductionSize + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    InitReduction<<<gridSize, blockSize>>>(d_flags, voxelCount, d_reduction, reductionSize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_reduction, d_reduction, reductionSize * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_reduction[i]);
    }

    // 释放内存
    free(h_flags);
    free(h_reduction);
    cudaFree(d_flags);
    cudaFree(d_reduction);

    return 0;
}
