#include <stdio.h>

// CUDA核函数
__global__ void cuda_record(float *p, float *seis_kt, int *Gxz, int ng) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < ng) {
        seis_kt[id] = p[Gxz[id]];
    }
}

int main() {
    // 设置数组大小
    const int array_size = 100;

    // 在设备上分配空间
    float *p_device, *seis_kt_device;
    int *Gxz_device;

    cudaMalloc((void**)&p_device, array_size * sizeof(float));
    cudaMalloc((void**)&seis_kt_device, array_size * sizeof(float));
    cudaMalloc((void**)&Gxz_device, array_size * sizeof(int));

    // 初始化输入数组数据
    float *p_host = (float *)malloc(array_size * sizeof(float));
    float *seis_kt_host = (float *)malloc(array_size * sizeof(float));
    int *Gxz_host = (int *)malloc(array_size * sizeof(int));

    for (int i = 0; i < array_size; ++i) {
        p_host[i] = i + 1.0; // 为了演示目的，将输入数据初始化为 1.0, 2.0, 3.0, ...
        seis_kt_host[i] = 0.0; // 初始化为0.0
        Gxz_host[i] = i % array_size; // 为了演示目的，将Gxz初始化为递增序列
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(p_device, p_host, array_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(seis_kt_device, seis_kt_host, array_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Gxz_device, Gxz_host, array_size * sizeof(int), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((array_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    cuda_record<<<gridDim, blockDim>>>(p_device, seis_kt_device, Gxz_device, array_size);

    // 将结果从设备复制回主机
    float *seis_kt_result = (float *)malloc(array_size * sizeof(float));
    cudaMemcpy(seis_kt_result, seis_kt_device, array_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < array_size; ++i) {
        printf("%.2f ", seis_kt_result[i]);
    }
    printf("\n");

    // 释放内存
    free(p_host);
    free(seis_kt_host);
    free(Gxz_host);
    free(seis_kt_result);
    cudaFree(p_device);
    cudaFree(seis_kt_device);
    cudaFree(Gxz_device);

    return 0;
}
 
