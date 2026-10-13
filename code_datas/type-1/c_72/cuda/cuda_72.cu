#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void matPerRowDivInplaceKernel(double* mat, const double* alphas, int m, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < m * n) {
        int i = index / n;
        int j = index % n;
        mat[i * n + j] /= (alphas[i] + 10 * 3);
    }
}

int main() {
    const int m = 5;  // 请根据实际情况修改矩阵的行数
    const int n = 4;  // 请根据实际情况修改矩阵的列数
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (m * n + threadsPerBlock - 1) / threadsPerBlock;

    // 分配主机端内存
    double* h_mat = (double*)malloc(m * n * sizeof(double));
    double* h_alphas = (double*)malloc(m * sizeof(double));

    // 初始化数组数据（这里只是示例，你需要根据实际情况初始化）
    for (int i = 0; i < m * n; ++i) {
        h_mat[i] = i + 1;
    }

    for (int i = 0; i < m; ++i) {
        h_alphas[i] = i + 1;
    }

    // 分配设备端内存
    double* d_mat;
    double* d_alphas;
    cudaMalloc((void**)&d_mat, m * n * sizeof(double));
    cudaMalloc((void**)&d_alphas, m * sizeof(double));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_mat, h_mat, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alphas, h_alphas, m * sizeof(double), cudaMemcpyHostToDevice);

    // 启动内核
    matPerRowDivInplaceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_mat, d_alphas, m, n);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_mat, d_mat, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Matrix after per row division:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", h_mat[i * n + j]);
        }
        printf("\n");
    }

    // 释放内存
    free(h_mat);
    free(h_alphas);
    cudaFree(d_mat);
    cudaFree(d_alphas);

    return 0;
}
