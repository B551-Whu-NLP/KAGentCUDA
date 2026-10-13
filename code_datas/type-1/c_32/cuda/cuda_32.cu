#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void intMultiply(int* result, const int* val1, const int val2, const unsigned int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        result[i] = val1[i] * val2;
    }
}

int main() {
    // 设置数组大小
    const int arraySize = 1000;

    // 分配主机端内存
    int* h_result = (int*)malloc(arraySize * sizeof(int));
    int* h_val1 = (int*)malloc(arraySize * sizeof(int));
    const int h_val2 = 2;  // 设置一个常量值

    // 初始化数据
    for (int i = 0; i < arraySize; ++i) {
        h_val1[i] = i;
    }

    // 分配设备端内存
    int* d_result;
    int* d_val1;
    cudaMalloc((void**)&d_result, arraySize * sizeof(int));
    cudaMalloc((void**)&d_val1, arraySize * sizeof(int));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_val1, h_val1, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    intMultiply<<<gridSize, blockSize>>>(d_result, d_val1, h_val2, arraySize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_result, d_result, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_result[i]);
    }

    // 释放内存
    free(h_result);
    free(h_val1);
    cudaFree(d_result);
    cudaFree(d_val1);

    return 0;
}
