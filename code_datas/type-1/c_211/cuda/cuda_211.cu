#include <stdio.h>

// CUDA核函数
__global__ void addV(int *a, int *b, int *c, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    // 设置数组大小
    const int array_size = 100;

    // 在设备上分配空间
    int *a_device, *b_device, *c_device;
    cudaMalloc((void**)&a_device, array_size * sizeof(int));
    cudaMalloc((void**)&b_device, array_size * sizeof(int));
    cudaMalloc((void**)&c_device, array_size * sizeof(int));

    // 初始化输入数组数据
    int *a_host = (int *)malloc(array_size * sizeof(int));
    int *b_host = (int *)malloc(array_size * sizeof(int));
    for (int i = 0; i < array_size; ++i) {
        a_host[i] = i + 1; // 为了演示目的，将输入数据初始化为 1, 2, 3, ...
        b_host[i] = (i + 1) * 2; // 为了演示目的，将输入数据初始化为 2, 4, 6, ...
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(a_device, a_host, array_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_host, array_size * sizeof(int), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((array_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    addV<<<gridDim, blockDim>>>(a_device, b_device, c_device, array_size);

    // 将结果从设备复制回主机
    int *c_result = (int *)malloc(array_size * sizeof(int));
    cudaMemcpy(c_result, c_device, array_size * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < array_size; ++i) {
        printf("%d ", c_result[i]);
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
 
