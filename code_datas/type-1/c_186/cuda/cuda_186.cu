#include <stdio.h>

// CUDA核函数
__global__ void allDivInplaceKernel(double *arr, double alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        arr[i] /= alpha;
    }
}

int main() {
    // 设置数据大小
    const int data_size = 100;
    const double alpha = 2.0;

    // 在设备上分配空间
    double *arr_device;
    cudaMalloc((void**)&arr_device, data_size * sizeof(double));

    // 初始化数据
    double *arr_host = (double *)malloc(data_size * sizeof(double));
    for (int i = 0; i < data_size; ++i) {
        arr_host[i] = i + 1.0; // 为了演示目的，将输入数据初始化为 1.0, 2.0, 3.0, ...
    }

    // 将数据从主机复制到设备
    cudaMemcpy(arr_device, arr_host, data_size * sizeof(double), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((data_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    allDivInplaceKernel<<<gridDim, blockDim>>>(arr_device, alpha, data_size);

    // 将结果从设备复制回主机
    cudaMemcpy(arr_host, arr_device, data_size * sizeof(double), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < data_size; ++i) {
        printf("%.2f ", arr_host[i]);
    }
    printf("\n");

    // 释放内存
    free(arr_host);
    cudaFree(arr_device);

    return 0;
}
 
