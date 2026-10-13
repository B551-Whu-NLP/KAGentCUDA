#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void fill_matrix(double* const A, const int rows, const int cols) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        A[row * cols + col] = static_cast<double>(row);
    }
}

int main() {
    // 定义矩阵大小
    const int rows = 10;
    const int cols = 5;

    // 分配主机端内存
    double* h_A = (double*)malloc(rows * cols * sizeof(double));

    // 分配设备端内存
    double* d_A;
    cudaMalloc((void**)&d_A, rows * cols * sizeof(double));

    // 设置线程块和网格大小
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    // 启动内核
    fill_matrix<<<gridSize, blockSize>>>(d_A, rows, cols);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_A, d_A, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < std::min(10, rows); ++i) {
        for (int j = 0; j < std::min(5, cols); ++j) {
            printf("%f ", h_A[i * cols + j]);
        }
        printf("\n");
    }

    // 释放内存
    free(h_A);
    cudaFree(d_A);

    return 0;
}
