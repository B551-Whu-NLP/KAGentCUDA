#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void Kernel_Sum_backward_opt2(float* db, float* sum, int r_sum, int c) {
    unsigned int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (j >= c)
        return;

    float temp = 0;

    for (int i = 0; i < r_sum; i++) {
        temp += sum[i * c + j];
    }

    db[j] = temp;
}

int main() {
    // 定义数组大小
    const int r_sum = 100;  // 请根据实际需求修改
    const int c = 50;       // 请根据实际需求修改

    // 分配主机端内存
    float* h_db = (float*)malloc(c * sizeof(float));
    float* h_sum = (float*)malloc(r_sum * c * sizeof(float));

    // 初始化数组数据
    for (int i = 0; i < r_sum * c; ++i) {
        h_sum[i] = static_cast<float>(i);  // Just an example, you can initialize it according to your needs
    }

    // 分配设备端内存
    float* d_db;
    float* d_sum;
    cudaMalloc((void**)&d_db, c * sizeof(float));
    cudaMalloc((void**)&d_sum, r_sum * c * sizeof(float));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_sum, h_sum, r_sum * c * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((c + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    Kernel_Sum_backward_opt2<<<gridSize, blockSize>>>(d_db, d_sum, r_sum, c);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_db, d_db, c * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("h_db[%d]: %f\n", i, h_db[i]);
    }

    // 释放内存
    free(h_db);
    free(h_sum);
    cudaFree(d_db);
    cudaFree(d_sum);

    return 0;
}
