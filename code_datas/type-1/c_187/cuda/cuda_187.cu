#include <stdio.h>

// CUDA核函数
__global__ void incrementArrayOnDevice(float *a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a[idx] = a[idx] + 1.0f;
    }
}

int main() {
    // 设置数据大小
    const int data_size = 100;

    // 在设备上分配空间
    float *a_device;
    cudaMalloc((void**)&a_device, data_size * sizeof(float));

    // 初始化数据
    float *a_host = (float *)malloc(data_size * sizeof(float));
    for (int i = 0; i < data_size; ++i) {
        a_host[i] = i + 1.0; // 为了演示目的，将输入数据初始化为 1.0, 2.0, 3.0, ...
    }

    // 将数据从主机复制到设备
    cudaMemcpy(a_device, a_host, data_size * sizeof(float), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((data_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    incrementArrayOnDevice<<<gridDim, blockDim>>>(a_device, data_size);

    // 将结果从设备复制回主机
    cudaMemcpy(a_host, a_device, data_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < data_size; ++i) {
        printf("%.2f ", a_host[i]);
    }
    printf("\n");

    // 释放内存
    free(a_host);
    cudaFree(a_device);

    return 0;
}
 
