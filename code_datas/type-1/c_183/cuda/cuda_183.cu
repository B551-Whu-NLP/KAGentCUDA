#include <stdio.h>

// CUDA核函数
__global__ void resetIndices(long *vec_out, const long N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        vec_out[idx] = idx;
    }
}

int main() {
    // 设置数据大小
    const long data_size = 100;

    // 在设备上分配空间
    long *vec_out_device;
    cudaMalloc((void**)&vec_out_device, data_size * sizeof(long));

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((data_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    resetIndices<<<gridDim, blockDim>>>(vec_out_device, data_size);

    // 将结果从设备复制回主机
    long *vec_out_host = (long *)malloc(data_size * sizeof(long));
    cudaMemcpy(vec_out_host, vec_out_device, data_size * sizeof(long), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (long i = 0; i < data_size; ++i) {
        printf("%ld ", vec_out_host[i]);
    }
    printf("\n");

    // 释放内存
    free(vec_out_host);
    cudaFree(vec_out_device);

    return 0;
}
 
