#include <stdio.h>

// CUDA核函数
__global__ void Copy_List(const int element_numbers, const float *origin_list, float *list) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < element_numbers) {
        list[i] = origin_list[i];
    }
}

int main() {
    // 设置数据大小
    const int data_size = 100;

    // 在设备上分配空间
    float *origin_list_device, *list_device;
    cudaMalloc((void**)&origin_list_device, data_size * sizeof(float));
    cudaMalloc((void**)&list_device, data_size * sizeof(float));

    // 初始化输入数据
    float *origin_list_host = (float *)malloc(data_size * sizeof(float));
    for (int i = 0; i < data_size; ++i) {
        origin_list_host[i] = i + 1.0; // 为了演示目的，将输入数据初始化为 1.0, 2.0, 3.0, ...
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(origin_list_device, origin_list_host, data_size * sizeof(float), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((data_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    Copy_List<<<gridDim, blockDim>>>(data_size, origin_list_device, list_device);

    // 将结果从设备复制回主机
    float *list_host = (float *)malloc(data_size * sizeof(float));
    cudaMemcpy(list_host, list_device, data_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < data_size; ++i) {
        printf("%.2f ", list_host[i]);
    }
    printf("\n");

    // 释放内存
    free(origin_list_host);
    free(list_host);
    cudaFree(origin_list_device);
    cudaFree(list_device);

    return 0;
}
 
