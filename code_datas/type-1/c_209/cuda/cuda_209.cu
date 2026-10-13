#include <stdio.h>

// CUDA核函数
__global__ void vecAddGPU(double *pdbA, double *pdbB, double *pdbC) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    pdbC[i] = pdbA[i] + pdbB[i];
}

int main() {
    // 设置向量大小
    const int vector_size = 100;

    // 在设备上分配空间
    double *pdbA_device, *pdbB_device, *pdbC_device;
    cudaMalloc((void**)&pdbA_device, vector_size * sizeof(double));
    cudaMalloc((void**)&pdbB_device, vector_size * sizeof(double));
    cudaMalloc((void**)&pdbC_device, vector_size * sizeof(double));

    // 初始化输入向量数据
    double *pdbA_host = (double *)malloc(vector_size * sizeof(double));
    double *pdbB_host = (double *)malloc(vector_size * sizeof(double));
    for (int i = 0; i < vector_size; ++i) {
        pdbA_host[i] = i + 1.0; // 为了演示目的，将输入数据初始化为 1.0, 2.0, 3.0, ...
        pdbB_host[i] = (i + 1.0) * 2.0; // 为了演示目的，将输入数据初始化为 2.0, 4.0, 6.0, ...
    }

    // 将输入数据从主机复制到设备
    cudaMemcpy(pdbA_device, pdbA_host, vector_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(pdbB_device, pdbB_host, vector_size * sizeof(double), cudaMemcpyHostToDevice);

    // 定义启动配置
    dim3 blockDim(256); // 块大小为256个线程
    dim3 gridDim((vector_size + blockDim.x - 1) / blockDim.x); // 确保足够的块数

    // 调用CUDA核函数
    vecAddGPU<<<gridDim, blockDim>>>(pdbA_device, pdbB_device, pdbC_device);

    // 将结果从设备复制回主机
    double *pdbC_result = (double *)malloc(vector_size * sizeof(double));
    cudaMemcpy(pdbC_result, pdbC_device, vector_size * sizeof(double), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result after CUDA kernel execution:\n");
    for (int i = 0; i < vector_size; ++i) {
        printf("%.2f ", pdbC_result[i]);
    }
    printf("\n");

    // 释放内存
    free(pdbA_host);
    free(pdbB_host);
    free(pdbC_result);
    cudaFree(pdbA_device);
    cudaFree(pdbB_device);
    cudaFree(pdbC_device);

    return 0;
}
 
