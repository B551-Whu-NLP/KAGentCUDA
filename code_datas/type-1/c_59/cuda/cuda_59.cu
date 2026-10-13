#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <stdio.h>

__global__ void forward_dropout_layer(float* input, int size, float* rand, float prob, float scale) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < size) {
        input[id] = (rand[id] < prob) ? 0 : input[id] * scale;
    }
}

int main() {
    // 定义数组大小
    const int size = 1000;

    // 分配主机端内存
    float* h_input = (float*)malloc(size * sizeof(float));
    float* h_rand = (float*)malloc(size * sizeof(float));

    // 初始化数组数据
    for (int i = 0; i < size; ++i) {
        h_input[i] = static_cast<float>(i);
        h_rand[i] = static_cast<float>(i) / size; // Random values between 0 and 1
    }

    // 分配设备端内存
    float* d_input;
    float* d_rand;
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_rand, size * sizeof(float));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rand, h_rand, size * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    forward_dropout_layer<<<gridSize, blockSize>>>(d_input, size, d_rand, 0.5f, 2.0f);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_input, d_input, size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("h_input[%d]: %f\n", i, h_input[i]);
    }

    // 释放内存
    free(h_input);
    free(h_rand);
    cudaFree(d_input);
    cudaFree(d_rand);

    return 0;
}
