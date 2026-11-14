#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void matVecRowSubInplaceKernel(double* mat, const double* vec, int m, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < m * n) {
        int i = index / n;
        int j = index % n;
        mat[i * n + j] -= vec[j];
    }
}

int main() {
    // 定义矩阵的行数和列数
    const int m = 5;
    const int n = 3;

    // 分配主机端内存
    double* h_mat = (double*)malloc(m * n * sizeof(double));
    double* h_vec = (double*)malloc(n * sizeof(double));

    // 初始化矩阵和向量数据
    for (int i = 0; i < m * n; ++i) {
        h_mat[i] = static_cast<double>(i); // Just an example, you can initialize it according to your needs
    }

    for (int i = 0; i < n; ++i) {
        h_vec[i] = static_cast<double>(i * 2); // Just an example, you can initialize it according to your needs
    }

    // 分配设备端内存
    double* d_mat;
    double* d_vec;
    cudaMalloc((void**)&d_mat, m * n * sizeof(double));
    cudaMalloc((void**)&d_vec, n * sizeof(double));

    // 将矩阵和向量数据从主机端拷贝到设备端
    cudaMemcpy(d_mat, h_mat, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec, n * sizeof(double), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((m * n + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    matVecRowSubInplaceKernel<<<gridSize, blockSize>>>(d_mat, d_vec, m, n);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_mat, d_mat, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("h_mat[%d][%d]: %f\n", i, j, h_mat[i * n + j]);
        }
    }

    // 释放内存
    free(h_mat);
    free(h_vec);
    cudaFree(d_mat);
    cudaFree(d_vec);

    return 0;
}
