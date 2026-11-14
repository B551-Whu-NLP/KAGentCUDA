#include <stdio.h>

// CUDA核函数
__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < len) {
        out[i] = in1[i] + in2[i];
    }
}

int main() {
    // 设置向量大小
    const int vector_size = 100;

    // 在设备上分配空间
    float *in1_device, *in2_device, *out_device;
    cudaMalloc((void**)&in1_device, vector_size * sizeof(float));
    cudaMalloc((void**)&in2_device, vector_size * sizeof(float));
    cudaMalloc((void**)&out_device, vector_size * sizeof(float));

    // 初始化输入向量数据
    float *in1_host = (float *)malloc(vector_size * sizeof(float));
    float *in2_host = (float *)malloc(vector_size * sizeof(float));
    for (int i = 0; i < vector_size; ++i) {
        in1_host[i] = i + 1.0; // 为了演示目的，将输入数据初始化为 1.0, 2.0, 3.0, ...
        in2_host[i] = (i + 1.0) * 2.0; // 为了演示目的，将输入数据初始化为 2.0, 4.0, 6.0, ...
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(in1_device, in1_host, vector_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(in2_device, in2_host, vector_size * sizeof(float), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((vector_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    vecAdd<<<gridDim, blockDim>>>(in1_device, in2_device, out_device, vector_size);

    // 将结果从设备复制回主机
    float *out_result = (float *)malloc(vector_size * sizeof(float));
    cudaMemcpy(out_result, out_device, vector_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < vector_size; ++i) {
        printf("%.2f ", out_result[i]);
    }
    printf("\n");

    // 释放内存
    free(in1_host);
    free(in2_host);
    free(out_result);
    cudaFree(in1_device);
    cudaFree(in2_device);
    cudaFree(out_device);

    return 0;
}
 
