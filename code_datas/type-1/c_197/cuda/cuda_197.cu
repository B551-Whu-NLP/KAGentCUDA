#include <stdio.h>

// CUDA核函数
__global__ void const_kernel(int N, float ALPHA, float *X, int INCX) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) {
        X[i * INCX] = ALPHA;
    }
}

int main() {
    // 设置数据大小
    const int data_size = 100;
    const float ALPHA = 2.0; // 为了演示目的，将ALPHA设为2.0

    // 在设备上分配空间
    float *X_device;
    cudaMalloc((void**)&X_device, data_size * sizeof(float));

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((data_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    const_kernel<<<gridDim, blockDim>>>(data_size, ALPHA, X_device, 1); // 假设INCX为1

    // 将结果从设备复制回主机
    float *X_host = (float *)malloc(data_size * sizeof(float));
    cudaMemcpy(X_host, X_device, data_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < data_size; ++i) {
        printf("%.2f ", X_host[i]);
    }
    printf("\n");

    // 释放内存
    free(X_host);
    cudaFree(X_device);

    return 0;
}
 
