#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void copy_swap(float* f_in, float* f_target, const int L_x) {
    const int k_x = threadIdx.x + blockIdx.x * blockDim.x;

    if (k_x < L_x) {
        float tempval = f_in[k_x];
        f_in[k_x] = f_target[k_x];
        f_target[k_x] = tempval;
    }
}

int main() {
    // 定义数组大小
    const int L_x = 1024;

    // 分配主机端内存
    float* h_f_in = (float*)malloc(L_x * sizeof(float));
    float* h_f_target = (float*)malloc(L_x * sizeof(float));

    // 初始化数组数据
    for (int i = 0; i < L_x; ++i) {
        h_f_in[i] = static_cast<float>(i); // Just an example, you can initialize it according to your needs
        h_f_target[i] = static_cast<float>(i * 2); // Just an example, you can initialize it according to your needs
    }

    // 分配设备端内存
    float* d_f_in;
    float* d_f_target;
    cudaMalloc((void**)&d_f_in, L_x * sizeof(float));
    cudaMalloc((void**)&d_f_target, L_x * sizeof(float));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_f_in, h_f_in, L_x * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_target, h_f_target, L_x * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((L_x + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    copy_swap<<<gridSize, blockSize>>>(d_f_in, d_f_target, L_x);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_f_in, d_f_in, L_x * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_f_target, d_f_target, L_x * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("h_f_in[%d]: %f, h_f_target[%d]: %f\n", i, h_f_in[i], i, h_f_target[i]);
    }

    // 释放内存
    free(h_f_in);
    free(h_f_target);
    cudaFree(d_f_in);
    cudaFree(d_f_target);

    return 0;
}
