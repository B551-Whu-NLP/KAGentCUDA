#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void memsetCudaInt(int* data, int val, int N) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < N) {
        data[index] = val;
    }
}

int main() {
    // 设置数组大小
    int arraySize = 1000;

    // 设置初始值
    int initialValue = 42;

    // 分配主机端内存
    int* h_data = (int*)malloc(arraySize * sizeof(int));

    // 分配设备端内存
    int* d_data;
    cudaMalloc((void**)&d_data, arraySize * sizeof(int));

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    memsetCudaInt<<<gridSize, blockSize>>>(d_data, initialValue, arraySize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_data, d_data, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_data[i]);
    }

    // 释放内存
    free(h_data);
    cudaFree(d_data);

    return 0;
}
