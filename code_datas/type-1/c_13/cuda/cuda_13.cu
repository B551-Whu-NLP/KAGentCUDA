#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void matDiagAddInplaceKernel(double* mat, double alpha, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        mat[i * dim + i] += alpha;
    }
}

int main() {
    // 设置矩阵维度
    int matrixDim = 5;

    // 分配主机端内存
    double* h_mat = (double*)malloc(matrixDim * matrixDim * sizeof(double));

    // 初始化数据
    for (int i = 0; i < matrixDim * matrixDim; ++i) {
        h_mat[i] = static_cast<double>(i);
    }

    // 分配设备端内存
    double* d_mat;
    cudaMalloc((void**)&d_mat, matrixDim * matrixDim * sizeof(double));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_mat, h_mat, matrixDim * matrixDim * sizeof(double), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (matrixDim * matrixDim + blockSize - 1) / blockSize;

    // 启动内核
    matDiagAddInplaceKernel<<<gridSize, blockSize>>>(d_mat, 2.0, matrixDim);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_mat, d_mat, matrixDim * matrixDim * sizeof(double), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < matrixDim; ++i) {
        for (int j = 0; j < matrixDim; ++j) {
            printf("%f ", h_mat[i * matrixDim + j]);
        }
        printf("\n");
    }

    // 释放内存
    free(h_mat);
    cudaFree(d_mat);

    return 0;
}
