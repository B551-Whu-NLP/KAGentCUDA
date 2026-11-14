#include <stdio.h>

// CUDA核函数
__global__ void clearArray(unsigned char *arr, const unsigned int length) {
    unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int skip = gridDim.x * blockDim.x;
    
    while (offset < length) {
        arr[offset] = 0;
        offset += skip;
    }
}

int main() {
    // 设置数据大小
    const unsigned int data_size = 100;

    // 在设备上分配空间
    unsigned char *arr_device;
    cudaMalloc((void**)&arr_device, data_size * sizeof(unsigned char));

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((data_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    clearArray<<<gridDim, blockDim>>>(arr_device, data_size);

    // 将结果从设备复制回主机
    unsigned char *arr_host = (unsigned char *)malloc(data_size * sizeof(unsigned char));
    cudaMemcpy(arr_host, arr_device, data_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (unsigned int i = 0; i < data_size; ++i) {
        printf("%d ", arr_host[i]);
    }
    printf("\n");

    // 释放内存
    free(arr_host);
    cudaFree(arr_device);

    return 0;
}
 
