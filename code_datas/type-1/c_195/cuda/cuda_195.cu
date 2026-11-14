#include <stdio.h>
#include <math.h>

// CUDA核函数
__global__ void sigmoid_kernel(float *input, float *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid] = 1.0 / (1.0 + expf(-input[tid]));
}

int main() {
    // 设置数据大小
    const int data_size = 100;

    // 在设备上分配空间
    float *input_device, *output_device;
    cudaMalloc((void**)&input_device, data_size * sizeof(float));
    cudaMalloc((void**)&output_device, data_size * sizeof(float));

    // 初始化输入数据
    float *input_host = (float *)malloc(data_size * sizeof(float));
    for (int i = 0; i < data_size; ++i) {
        input_host[i] = i + 1.0; // 为了演示目的，将输入数据初始化为 1.0, 2.0, 3.0, ...
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(input_device, input_host, data_size * sizeof(float), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((data_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    sigmoid_kernel<<<gridDim, blockDim>>>(input_device, output_device);

    // 将结果从设备复制回主机
    float *output_host = (float *)malloc(data_size * sizeof(float));
    cudaMemcpy(output_host, output_device, data_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < data_size; ++i) {
        printf("%.6f ", output_host[i]);
    }
    printf("\n");

    // 释放内存
    free(input_host);
    free(output_host);
    cudaFree(input_device);
    cudaFree(output_device);

    return 0;
}
 
