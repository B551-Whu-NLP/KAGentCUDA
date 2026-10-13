#include <stdio.h>

// CUDA核函数
__global__ void k_vec_divide(float *vec1, float *vec2, size_t max_size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < max_size; i += blockDim.x * gridDim.x) {
        vec1[i] = vec1[i] / vec2[i];
    }
}

int main() {
    // 设置数组大小
    const size_t array_size = 100;

    // 在设备上分配空间
    float *vec1_device, *vec2_device;
    cudaMalloc((void**)&vec1_device, array_size * sizeof(float));
    cudaMalloc((void**)&vec2_device, array_size * sizeof(float));

    // 初始化输入数组数据
    float *vec1_host = (float *)malloc(array_size * sizeof(float));
    float *vec2_host = (float *)malloc(array_size * sizeof(float));
    for (size_t i = 0; i < array_size; ++i) {
        vec1_host[i] = i + 1.0; // 为了演示目的，将输入数据初始化为 1.0, 2.0, 3.0, ...
        vec2_host[i] = (i + 1.0) * 2.0; // 为了演示目的，将输入数据初始化为 2.0, 4.0, 6.0, ...
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(vec1_device, vec1_host, array_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vec2_device, vec2_host, array_size * sizeof(float), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((array_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    k_vec_divide<<<gridDim, blockDim>>>(vec1_device, vec2_device, array_size);

    // 将结果从设备复制回主机
    float *vec1_result = (float *)malloc(array_size * sizeof(float));
    cudaMemcpy(vec1_result, vec1_device, array_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (size_t i = 0; i < array_size; ++i) {
        printf("%.2f ", vec1_result[i]);
    }
    printf("\n");

    // 释放内存
    free(vec1_host);
    free(vec2_host);
    free(vec1_result);
    cudaFree(vec1_device);
    cudaFree(vec2_device);

    return 0;
}
 
