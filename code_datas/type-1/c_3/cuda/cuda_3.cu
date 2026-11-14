#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void square(int* array, int arrayCount) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < arrayCount) {
        array[idx] *= array[idx];
    }
}

int main() {
    // 设置数组大小
    int arrayCount = 1000;

    // 分配主机端内存
    int* h_array = (int*)malloc(arrayCount * sizeof(int));

    // 初始化数据
    for (int i = 0; i < arrayCount; ++i) {
        h_array[i] = i;
    }

    // 分配设备端内存
    int* d_array;
    cudaMalloc((void**)&d_array, arrayCount * sizeof(int));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_array, h_array, arrayCount * sizeof(int), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arrayCount + blockSize - 1) / blockSize;

    // 启动内核
    square<<<gridSize, blockSize>>>(d_array, arrayCount);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_array, d_array, arrayCount * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_array[i]);
    }

    // 释放内存
    free(h_array);
    cudaFree(d_array);

    return 0;
}
