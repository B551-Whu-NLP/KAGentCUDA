#include <stdio.h>

// CUDA核函数
__global__ void setLabels(int *output, int dims, int clsNum) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dims) {
        return;
    }
    output[tid] = tid % clsNum;
}

int main() {
    // 设置数据大小
    const int data_size = 100;
    const int clsNum = 5;

    // 在设备上分配空间
    int *output_device;
    cudaMalloc((void**)&output_device, data_size * sizeof(int));

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((data_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    setLabels<<<gridDim, blockDim>>>(output_device, data_size, clsNum);

    // 将结果从设备复制回主机
    int *output_host = (int *)malloc(data_size * sizeof(int));
    cudaMemcpy(output_host, output_device, data_size * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < data_size; ++i) {
        printf("%d ", output_host[i]);
    }
    printf("\n");

    // 释放内存
    free(output_host);
    cudaFree(output_device);

    return 0;
}
 
