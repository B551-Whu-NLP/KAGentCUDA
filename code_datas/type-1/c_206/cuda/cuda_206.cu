#include <stdio.h>

// CUDA核函数
__global__ void doubleArrayScalarAddKernel(double *d_in, double *d_out, int length, double scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < length) {
        d_out[tid] = d_in[tid] + scalar;
    }
}

int main() {
    // 设置数组大小
    const int array_size = 100;

    // 在设备上分配空间
    double *d_in_device, *d_out_device;
    cudaMalloc((void**)&d_in_device, array_size * sizeof(double));
    cudaMalloc((void**)&d_out_device, array_size * sizeof(double));

    // 初始化输入数组数据
    double *d_in_host = (double *)malloc(array_size * sizeof(double));
    for (int i = 0; i < array_size; ++i) {
        d_in_host[i] = i + 1.0; // 为了演示目的，将输入数据初始化为 1.0, 2.0, 3.0, ...
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(d_in_device, d_in_host, array_size * sizeof(double), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((array_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 定义标量值
    double scalar = 5.0;

    // 调用CUDA核函数
    doubleArrayScalarAddKernel<<<gridDim, blockDim>>>(d_in_device, d_out_device, array_size, scalar);

    // 将结果从设备复制回主机
    double *d_out_result = (double *)malloc(array_size * sizeof(double));
    cudaMemcpy(d_out_result, d_out_device, array_size * sizeof(double), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < array_size; ++i) {
        printf("%.2f ", d_out_result[i]);
    }
    printf("\n");

    // 释放内存
    free(d_in_host);
    free(d_out_result);
    cudaFree(d_in_device);
    cudaFree(d_out_device);

    return 0;
}
 
