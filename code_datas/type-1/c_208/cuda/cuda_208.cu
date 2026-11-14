#include <stdio.h>

// CUDA核函数
__global__ void test1(float *input, int dims) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dims) {
        return;
    }
    if (input[tid * 4] != 0) {
        input[tid * 4] = 0;
    }
}

int main() {
    // 设置数组大小
    const int array_size = 100;

    // 在设备上分配空间
    float *input_device;
    cudaMalloc((void**)&input_device, array_size * 4 * sizeof(float));

    // 初始化输入数组数据
    float *input_host = (float *)malloc(array_size * 4 * sizeof(float));
    for (int i = 0; i < array_size * 4; ++i) {
        input_host[i] = i + 1.0; // 为了演示目的，将输入数据初始化为 1.0, 2.0, 3.0, ...
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(input_device, input_host, array_size * 4 * sizeof(float), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((array_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    test1<<<gridDim, blockDim>>>(input_device, array_size);

    // 将结果从设备复制回主机
    float *output_result = (float *)malloc(array_size * 4 * sizeof(float));
    cudaMemcpy(output_result, input_device, array_size * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < array_size * 4; ++i) {
        printf("%.2f ", output_result[i]);
    }
    printf("\n");

    // 释放内存
    free(input_host);
    free(output_result);
    cudaFree(input_device);

    return 0;
}
 
