#include <stdio.h>

// CUDA核函数
__global__ void subAvg(int *input, int count, int avg) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < count) {
        input[index] = input[index] - avg;
    }
}

int main() {
    // 设置数据大小
    const int data_size = 100;

    // 在设备上分配空间
    int *input_device;
    cudaMalloc((void**)&input_device, data_size * sizeof(int));

    // 初始化输入数据
    int *input_host = (int *)malloc(data_size * sizeof(int));
    for (int i = 0; i < data_size; ++i) {
        input_host[i] = i + 1; // 为了演示目的，将输入数据初始化为 1, 2, 3, ...
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(input_device, input_host, data_size * sizeof(int), cudaMemcpyHostToDevice);

    // 计算平均值
    int sum = 0;
    for (int i = 0; i < data_size; ++i) {
        sum += input_host[i];
    }
    int avg = sum / data_size;

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((data_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    subAvg<<<gridDim, blockDim>>>(input_device, data_size, avg);

    // 将结果从设备复制回主机
    cudaMemcpy(input_host, input_device, data_size * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < data_size; ++i) {
        printf("%d ", input_host[i]);
    }
    printf("\n");

    // 释放内存
    free(input_host);
    cudaFree(input_device);

    return 0;
}
 
