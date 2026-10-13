#include <stdio.h>

// CUDA核函数
__global__ void Init(const long long size, const double *in, double *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = in[i];
    }
}

int main() {
    // 设置数据大小
    const long long data_size = 100;

    // 在设备上分配空间
    double *in_device, *out_device;
    cudaMalloc((void**)&in_device, data_size * sizeof(double));
    cudaMalloc((void**)&out_device, data_size * sizeof(double));

    // 初始化输入数据
    double *in_host = (double *)malloc(data_size * sizeof(double));
    for (long long i = 0; i < data_size; ++i) {
        in_host[i] = i + 1.0; // 为了演示目的，将输入数据初始化为 1.0, 2.0, 3.0, ...
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(in_device, in_host, data_size * sizeof(double), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((data_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    Init<<<gridDim, blockDim>>>(data_size, in_device, out_device);

    // 将结果从设备复制回主机
    double *out_host = (double *)malloc(data_size * sizeof(double));
    cudaMemcpy(out_host, out_device, data_size * sizeof(double), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (long long i = 0; i < data_size; ++i) {
        printf("%.2f ", out_host[i]);
    }
    printf("\n");

    // 释放内存
    free(in_host);
    free(out_host);
    cudaFree(in_device);
    cudaFree(out_device);

    return 0;
}
 
