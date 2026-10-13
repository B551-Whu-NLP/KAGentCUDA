#include <stdio.h>

// CUDA核函数
__global__ void kmeans_set_zero(int *means) {
    means[blockIdx.x * blockDim.x + threadIdx.x] = 0;
}

int main() {
    // 设置数据大小
    const int data_size = 100;

    // 分配和初始化数据
    int *means_host = (int *)malloc(data_size * sizeof(int));
    for (int i = 0; i < data_size; ++i) {
        means_host[i] = i;
    }

    // 在设备上分配空间并将数据复制到设备
    int *means_device;
    cudaMalloc((void**)&means_device, data_size * sizeof(int));
    cudaMemcpy(means_device, means_host, data_size * sizeof(int), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((data_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    kmeans_set_zero<<<gridDim, blockDim>>>(means_device);

    // 将结果从设备复制回主机
    cudaMemcpy(means_host, means_device, data_size * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < data_size; ++i) {
        printf("%d ", means_host[i]);
    }
    printf("\n");

    // 释放内存
    free(means_host);
    cudaFree(means_device);

    return 0;
}
 
