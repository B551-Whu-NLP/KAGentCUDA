#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void countRangesGlobal(int size, int* A, int* B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;

    int x = A[i] / 100;
    atomicAdd(&B[x], 1);
}

int main() {
    // 设置数组大小
    int arraySize = 1000;

    // 分配主机端内存
    int* h_A = (int*)malloc(arraySize * sizeof(int));
    int* h_B = (int*)malloc(arraySize * sizeof(int));

    // 初始化数据
    for (int i = 0; i < arraySize; ++i) {
        h_A[i] = i;
        h_B[i] = 0;
    }

    // 分配设备端内存
    int* d_A;
    int* d_B;
    cudaMalloc((void**)&d_A, arraySize * sizeof(int));
    cudaMalloc((void**)&d_B, arraySize * sizeof(int));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_A, h_A, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    countRangesGlobal<<<gridSize, blockSize>>>(arraySize, d_A, d_B);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_B, d_B, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_B[i]);
    }

    // 释放内存
    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
