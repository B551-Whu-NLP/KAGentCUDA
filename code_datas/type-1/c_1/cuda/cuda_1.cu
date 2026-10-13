#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void add_100(int numElements, int* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numElements) {
        data[tid] += 100;
    }
}

int main() {
    // 设置数据大小
    int numElements = 1000;

    // 分配主机端内存
    int* h_data = (int*)malloc(numElements * sizeof(int));

    // 初始化数据
    for (int i = 0; i < numElements; ++i) {
        h_data[i] = i;
    }

    // 分配设备端内存
    int* d_data;
    cudaMalloc((void**)&d_data, numElements * sizeof(int));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_data, h_data, numElements * sizeof(int), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;

    // 启动内核
    add_100<<<gridSize, blockSize>>>(numElements, d_data);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_data, d_data, numElements * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    for (int i = 0; i < numElements; ++i) {
        printf("%d ", h_data[i]);
    }

    // 释放内存
    free(h_data);
    cudaFree(d_data);

    return 0;
}
