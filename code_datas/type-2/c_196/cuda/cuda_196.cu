#include <stdio.h>

// CUDA核函数
__global__ void kernelUpdateHead(int *head, int *d_idxs_out, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        head[d_idxs_out[i]] = 1;
    }
}

int main() {
    // 设置数据大小
    const int data_size = 100;

    // 在设备上分配空间
    int *head_device, *d_idxs_out_device;
    cudaMalloc((void**)&head_device, data_size * sizeof(int));
    cudaMalloc((void**)&d_idxs_out_device, data_size * sizeof(int));

    // 初始化输入数据
    int *d_idxs_out_host = (int *)malloc(data_size * sizeof(int));
    for (int i = 0; i < data_size; ++i) {
        d_idxs_out_host[i] = i % data_size; // 为了演示目的，将索引值初始化为0到data_size-1的循环
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(d_idxs_out_device, d_idxs_out_host, data_size * sizeof(int), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((data_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    kernelUpdateHead<<<gridDim, blockDim>>>(head_device, d_idxs_out_device, data_size);

    // 将结果从设备复制回主机
    int *head_host = (int *)malloc(data_size * sizeof(int));
    cudaMemcpy(head_host, head_device, data_size * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < data_size; ++i) {
        printf("%d ", head_host[i]);
    }
    printf("\n");

    // 释放内存
    free(d_idxs_out_host);
    free(head_host);
    cudaFree(d_idxs_out_device);
    cudaFree(head_device);

    return 0;
}
 
