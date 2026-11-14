#include <stdio.h>

// CUDA核函数
__global__ void add_matrix(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // 设置矩阵大小
    const int matrix_size = 100;

    // 在设备上分配空间
    float *a_device, *b_device, *c_device;
    cudaMalloc((void**)&a_device, matrix_size * sizeof(float));
    cudaMalloc((void**)&b_device, matrix_size * sizeof(float));
    cudaMalloc((void**)&c_device, matrix_size * sizeof(float));

    // 初始化输入矩阵数据
    float *a_host = (float *)malloc(matrix_size * sizeof(float));
    float *b_host = (float *)malloc(matrix_size * sizeof(float));
    for (int i = 0; i < matrix_size; ++i) {
        a_host[i] = i + 1.0; // 为了演示目的，将输入数据初始化为 1.0, 2.0, 3.0, ...
        b_host[i] = (i + 1.0) * 2.0; // 为了演示目的，将输入数据初始化为 2.0, 4.0, 6.0, ...
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(a_device, a_host, matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_host, matrix_size * sizeof(float), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((matrix_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    add_matrix<<<gridDim, blockDim>>>(a_device, b_device, c_device, matrix_size);

    // 将结果从设备复制回主机
    float *c_result = (float *)malloc(matrix_size * sizeof(float));
    cudaMemcpy(c_result, c_device, matrix_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < matrix_size; ++i) {
        printf("%.2f ", c_result[i]);
    }
    printf("\n");

    // 释放内存
    free(a_host);
    free(b_host);
    free(c_result);
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);

    return 0;
}
 
