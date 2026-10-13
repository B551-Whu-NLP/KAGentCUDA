#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void pathPlan(int* devSpeed, int* devSteer, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size) {
        devSpeed[tid] += 1;
        devSteer[tid] += 1;
        tid += blockDim.x * gridDim.x;
    }
}

int main() {
    // 定义数组大小
    const int arraySize = 1000;

    // 分配主机端内存
    int* h_devSpeed = (int*)malloc(arraySize * sizeof(int));
    int* h_devSteer = (int*)malloc(arraySize * sizeof(int));

    // 初始化数组数据
    for (int i = 0; i < arraySize; ++i) {
        h_devSpeed[i] = i;
        h_devSteer[i] = 2 * i;
    }

    // 分配设备端内存
    int* d_devSpeed;
    int* d_devSteer;
    cudaMalloc((void**)&d_devSpeed, arraySize * sizeof(int));
    cudaMalloc((void**)&d_devSteer, arraySize * sizeof(int));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_devSpeed, h_devSpeed, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_devSteer, h_devSteer, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    pathPlan<<<gridSize, blockSize>>>(d_devSpeed, d_devSteer, arraySize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_devSpeed, d_devSpeed, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_devSteer, d_devSteer, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("Speed: %d, Steer: %d\n", h_devSpeed[i], h_devSteer[i]);
    }

    // 释放内存
    free(h_devSpeed);
    free(h_devSteer);
    cudaFree(d_devSpeed);
    cudaFree(d_devSteer);

    return 0;
}
