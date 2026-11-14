#include <stdio.h>

// CUDA核函数
__global__ void histogram(int n, int *color, int *bucket) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        int c = color[i];
        atomicAdd(&bucket[c], 1);
    }
}

int main() {
    // 设置数据大小
    const int data_size = 100;
    const int num_bins = 256; // 假设颜色值范围为0-255

    // 在设备上分配空间
    int *color_device, *bucket_device;
    cudaMalloc((void**)&color_device, data_size * sizeof(int));
    cudaMalloc((void**)&bucket_device, num_bins * sizeof(int));

    // 初始化输入数据
    int *color_host = (int *)malloc(data_size * sizeof(int));
    for (int i = 0; i < data_size; ++i) {
        color_host[i] = i % num_bins; // 为了演示目的，将颜色值初始化为0-255的循环
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(color_device, color_host, data_size * sizeof(int), cudaMemcpyHostToDevice);

    // 初始化输出数据
    int *bucket_host = (int *)malloc(num_bins * sizeof(int));
    memset(bucket_host, 0, num_bins * sizeof(int)); // 初始化为零

    // 将输出数据从主机复制到设备
    cudaMemcpy(bucket_device, bucket_host, num_bins * sizeof(int), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((data_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    histogram<<<gridDim, blockDim>>>(data_size, color_device, bucket_device);

    // 将结果从设备复制回主机
    cudaMemcpy(bucket_host, bucket_device, num_bins * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Histogram Result after CUDA kernel execution:\n");
    for (int i = 0; i < num_bins; ++i) {
        printf("Bucket %d: %d\n", i, bucket_host[i]);
    }

    // 释放内存
    free(color_host);
    free(bucket_host);
    cudaFree(color_device);
    cudaFree(bucket_device);

    return 0;
}
 
