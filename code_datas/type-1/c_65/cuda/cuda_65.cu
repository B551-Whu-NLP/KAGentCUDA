#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void MMDOuterProdComputeWithSum(float* x_average, int size_x, float* x_outer_prod) {
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;

    for (int i = block_id * blockDim.x + thread_id; i < size_x; i += gridDim.x * blockDim.x) {
        x_outer_prod[i] = x_average[i] * x_average[i];
    }
}

int main() {
    // 定义数组大小
    const int size_x = 100;

    // 分配主机端内存
    float* h_x_average = (float*)malloc(size_x * sizeof(float));
    float* h_x_outer_prod = (float*)malloc(size_x * sizeof(float));

    // 初始化数组数据
    for (int i = 0; i < size_x; ++i) {
        h_x_average[i] = static_cast<float>(i); // Just an example, you can initialize it according to your needs
    }

    // 分配设备端内存
    float* d_x_average;
    float* d_x_outer_prod;
    cudaMalloc((void**)&d_x_average, size_x * sizeof(float));
    cudaMalloc((void**)&d_x_outer_prod, size_x * sizeof(float));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_x_average, h_x_average, size_x * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((size_x + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    MMDOuterProdComputeWithSum<<<gridSize, blockSize>>>(d_x_average, size_x, d_x_outer_prod);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_x_outer_prod, d_x_outer_prod, size_x * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < size_x; ++i) {
        printf("h_x_outer_prod[%d]: %f\n", i, h_x_outer_prod[i]);
    }

    // 释放内存
    free(h_x_average);
    free(h_x_outer_prod);
    cudaFree(d_x_average);
    cudaFree(d_x_outer_prod);

    return 0;
}
