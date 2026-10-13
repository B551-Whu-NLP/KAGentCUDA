#include <stdio.h>

// CUDA核函数
__global__ void Mul_half(float *src, float *dst) {
    int index = threadIdx.x;
    if (index < 3) {
        dst[index] = src[index] * 0.5;
    }
}

int main() {
    // 设置数据大小
    const int data_size = 3;

    // 分配和初始化数据
    float *src_host = (float *)malloc(data_size * sizeof(float));
    for (int i = 0; i < data_size; ++i) {
        src_host[i] = i + 1.0; // 为了演示目的，将输入数据初始化为 1.0, 2.0, 3.0
    }

    // 在设备上分配空间并将数据复制到设备
    float *src_device, *dst_device;
    cudaMalloc((void**)&src_device, data_size * sizeof(float));
    cudaMalloc((void**)&dst_device, data_size * sizeof(float));
    cudaMemcpy(src_device, src_host, data_size * sizeof(float), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim(1);    // 一个块足够

    // 调用CUDA核函数
    Mul_half<<<gridDim, blockDim>>>(src_device, dst_device);

    // 将结果从设备复制回主机
    float *dst_host = (float *)malloc(data_size * sizeof(float));
    cudaMemcpy(dst_host, dst_device, data_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < data_size; ++i) {
        printf("%.2f ", dst_host[i]);
    }
    printf("\n");

    // 释放内存
    free(src_host);
    free(dst_host);
    cudaFree(src_device);
    cudaFree(dst_device);

    return 0;
}
 
