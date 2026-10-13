#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void is_repeat(int N, int* device_input, int* device_output) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N) {
        device_output[idx] = 0;

        if (idx + 1 < N && device_input[idx] == device_input[idx + 1])
            device_output[idx] = 1;
    }
}

int main() {
    // 定义数组大小
    const int N = 100;  // 请根据实际需求修改

    // 分配主机端内存
    int* h_device_input = (int*)malloc(N * sizeof(int));
    int* h_device_output = (int*)malloc(N * sizeof(int));

    // 初始化数组数据
    for (int i = 0; i < N; ++i) {
        h_device_input[i] = i % 10;  // Just an example, you can initialize it according to your needs
    }

    // 分配设备端内存
    int* d_device_input;
    int* d_device_output;
    cudaMalloc((void**)&d_device_input, N * sizeof(int));
    cudaMalloc((void**)&d_device_output, N * sizeof(int));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_device_input, h_device_input, N * sizeof(int), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    is_repeat<<<gridSize, blockSize>>>(N, d_device_input, d_device_output);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_device_output, d_device_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < N; ++i) {
        printf("h_device_output[%d]: %d\n", i, h_device_output[i]);
    }

    // 释放内存
    free(h_device_input);
    free(h_device_output);
    cudaFree(d_device_input);
    cudaFree(d_device_output);

    return 0;
}
