#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void matColMeanDiv(double* buf, int m, int n, double* tmp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        buf[i] = tmp[i] / static_cast<double>(m);
    }
}

int main() {
    // 设置矩阵大小
    int rows = 1000;
    int cols = 10;

    // 分配主机端内存
    double* h_buf = (double*)malloc(cols * sizeof(double));
    double* h_tmp = (double*)malloc(cols * sizeof(double));

    // 初始化数据
    for (int i = 0; i < cols; ++i) {
        h_tmp[i] = static_cast<double>(i);
    }

    // 分配设备端内存
    double* d_buf;
    double* d_tmp;
    cudaMalloc((void**)&d_buf, cols * sizeof(double));
    cudaMalloc((void**)&d_tmp, cols * sizeof(double));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_tmp, h_tmp, cols * sizeof(double), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (cols + blockSize - 1) / blockSize;

    // 启动内核
    matColMeanDiv<<<gridSize, blockSize>>>(d_buf, rows, cols, d_tmp);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_buf, d_buf, cols * sizeof(double), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_buf[i]);
    }

    // 释放内存
    free(h_buf);
    free(h_tmp);
    cudaFree(d_buf);
    cudaFree(d_tmp);

    return 0;
}
