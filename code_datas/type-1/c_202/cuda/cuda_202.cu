#include <stdio.h>

// CUDA核函数
__global__ void gpu_add(float *c, float *a, float *b, int n) {
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        c[k] = a[k] + b[k];
    }
}

int main() {
    // 设置数据大小
    const int data_size = 100;

    // 在设备上分配空间
    float *a_device, *b_device, *c_device;
    cudaMalloc((void**)&a_device, data_size * sizeof(float));
    cudaMalloc((void**)&b_device, data_size * sizeof(float));
    cudaMalloc((void**)&c_device, data_size * sizeof(float));

    // 初始化输入数据
    float *a_host = (float *)malloc(data_size * sizeof(float));
    float *b_host = (float *)malloc(data_size * sizeof(float));
    for (int i = 0; i < data_size; ++i) {
        a_host[i] = i + 1.0; // 为了演示目的，将输入数据初始化为 1.0, 2.0, 3.0, ...
        b_host[i] = (i + 1.0) * 2.0; // 为了演示目的，将输入数据初始化为 2.0, 4.0, 6.0, ...
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(a_device, a_host, data_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_host, data_size * sizeof(float), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim(1); // 只使用一个块

    // 调用CUDA核函数
    gpu_add<<<gridDim, blockDim>>>(c_device, a_device, b_device, data_size);

    // 将结果从设备复制回主机
    float *c_result = (float *)malloc(data_size * sizeof(float));
    cudaMemcpy(c_result, c_device, data_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < data_size; ++i) {
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
 
