#include <stdio.h>

// CUDA核函数
__global__ void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = index; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    // 设置数据大小
    const int data_size = 100;

    // 在设备上分配空间
    float *x_device, *y_device;
    cudaMalloc((void**)&x_device, data_size * sizeof(float));
    cudaMalloc((void**)&y_device, data_size * sizeof(float));

    // 初始化输入数据
    float *x_host = (float *)malloc(data_size * sizeof(float));
    float *y_host = (float *)malloc(data_size * sizeof(float));
    for (int i = 0; i < data_size; ++i) {
        x_host[i] = i + 1.0; // 为了演示目的，将输入数据初始化为 1.0, 2.0, 3.0, ...
        y_host[i] = (i + 1.0) * 2.0; // 为了演示目的，将输入数据初始化为 2.0, 4.0, 6.0, ...
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(x_device, x_host, data_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_host, data_size * sizeof(float), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((data_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    add<<<gridDim, blockDim>>>(data_size, x_device, y_device);

    // 将结果从设备复制回主机
    float *y_result = (float *)malloc(data_size * sizeof(float));
    cudaMemcpy(y_result, y_device, data_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < data_size; ++i) {
        printf("%.2f ", y_result[i]);
    }
    printf("\n");

    // 释放内存
    free(x_host);
    free(y_host);
    free(y_result);
    cudaFree(x_device);
    cudaFree(y_device);

    return 0;
}
 
