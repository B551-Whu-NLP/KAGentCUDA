#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void transposeNaive(int* vector, int* transposed, int size) {
    int column = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < size && column < size) {
        transposed[row + column * size] = vector[column + row * size];
    }
}

int main() {
    // 定义矩阵大小
    const int size = 4;

    // 分配主机端内存
    int* h_vector = (int*)malloc(size * size * sizeof(int));
    int* h_transposed = (int*)malloc(size * size * sizeof(int));

    // 初始化矩阵数据
    for (int i = 0; i < size * size; ++i) {
        h_vector[i] = i;
    }

    // 分配设备端内存
    int* d_vector;
    int* d_transposed;
    cudaMalloc((void**)&d_vector, size * size * sizeof(int));
    cudaMalloc((void**)&d_transposed, size * size * sizeof(int));

    // 将矩阵数据从主机端拷贝到设备端
    cudaMemcpy(d_vector, h_vector, size * size * sizeof(int), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(2, 2);  // 2x2 线程块
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x, (size + blockSize.y - 1) / blockSize.y);

    // 启动内核
    transposeNaive<<<gridSize, blockSize>>>(d_vector, d_transposed, size);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_transposed, d_transposed, size * size * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印原始矩阵
    printf("Original Matrix:\n");
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf("%d\t", h_vector[j + i * size]);
        }
        printf("\n");
    }

    // 打印转置矩阵
    printf("\nTransposed Matrix:\n");
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf("%d\t", h_transposed[j + i * size]);
        }
        printf("\n");
    }

    // 释放内存
    free(h_vector);
    free(h_transposed);
    cudaFree(d_vector);
    cudaFree(d_transposed);

    return 0;
}
