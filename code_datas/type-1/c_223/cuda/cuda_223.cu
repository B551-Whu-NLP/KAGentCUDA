#include <stdio.h>

// CUDA核函数
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // 设置数组大小
    const int array_size = 100;

    // 在设备上分配空间
    float *A_device, *B_device, *C_device;

    cudaMalloc((void**)&A_device, array_size * sizeof(float));
    cudaMalloc((void**)&B_device, array_size * sizeof(float));
    cudaMalloc((void**)&C_device, array_size * sizeof(float));

    // 初始化输入数组数据
    float *A_host = (float *)malloc(array_size * sizeof(float));
    float *B_host = (float *)malloc(array_size * sizeof(float));

    for (int i = 0; i < array_size; ++i) {
        A_host[i] = i + 1.0; // 为了演示目的，将输入数据初始化为 1.0, 2.0, 3.0, ...
        B_host[i] = (i + 1.0) * 2.0; // 为了演示目的，将输入数据初始化为 2.0, 4.0, 6.0, ...
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(A_device, A_host, array_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, array_size * sizeof(float), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((array_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    vectorAdd<<<gridDim, blockDim>>>(A_device, B_device, C_device, array_size);

    // 将结果从设备复制回主机
    float *C_host = (float *)malloc(array_size * sizeof(float));
    cudaMemcpy(C_host, C_device, array_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < array_size; ++i) {
        printf("%.2f ", C_host[i]);
    }
    printf("\n");

    // 释放内存
    free(A_host);
    free(B_host);
    free(C_host);
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    return 0;
}
 
