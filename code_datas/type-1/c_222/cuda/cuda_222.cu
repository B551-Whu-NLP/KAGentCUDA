#include <stdio.h>

// CUDA核函数
__global__ void kernelSAXPY(int len, float a, float *d_x, float *d_y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        d_y[i] = d_x[i] * a + d_y[i];
    }
}

int main() {
    // 设置数组大小
    const int array_size = 100;

    // 在设备上分配空间
    float *d_x, *d_y;

    cudaMalloc((void**)&d_x, array_size * sizeof(float));
    cudaMalloc((void**)&d_y, array_size * sizeof(float));

    // 初始化输入数组数据
    float *h_x = (float *)malloc(array_size * sizeof(float));
    float *h_y = (float *)malloc(array_size * sizeof(float));

    for (int i = 0; i < array_size; ++i) {
        h_x[i] = i + 1.0; // 为了演示目的，将输入数据初始化为 1.0, 2.0, 3.0, ...
        h_y[i] = (i + 1.0) * 2.0; // 为了演示目的，将输入数据初始化为 2.0, 4.0, 6.0, ...
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(d_x, h_x, array_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, array_size * sizeof(float), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((array_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 定义标量值
    float a = 0.5;

    // 调用CUDA核函数
    kernelSAXPY<<<gridDim, blockDim>>>(array_size, a, d_x, d_y);

    // 将结果从设备复制回主机
    float *h_result = (float *)malloc(array_size * sizeof(float));
    cudaMemcpy(h_result, d_y, array_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < array_size; ++i) {
        printf("%.2f ", h_result[i]);
    }
    printf("\n");

    // 释放内存
    free(h_x);
    free(h_y);
    free(h_result);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
 
