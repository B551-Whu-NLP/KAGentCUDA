#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void delay_kernel(int* N_mobil, int* Tau, int dia) {
    int N = N_mobil[0];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        if (Tau[id] > 0) {
            Tau[id] = Tau[id] - 1;
        }
    }
}

int main() {
    // 定义数组大小
    const int N = 1000;

    // 分配主机端内存
    int* h_N_mobil = (int*)malloc(sizeof(int));
    int* h_Tau = (int*)malloc(N * sizeof(int));

    // 初始化数组数据
    h_N_mobil[0] = N;
    for (int i = 0; i < N; ++i) {
        h_Tau[i] = static_cast<int>(i);
    }

    // 分配设备端内存
    int* d_N_mobil;
    int* d_Tau;
    cudaMalloc((void**)&d_N_mobil, sizeof(int));
    cudaMalloc((void**)&d_Tau, N * sizeof(int));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_N_mobil, h_N_mobil, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Tau, h_Tau, N * sizeof(int), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    delay_kernel<<<gridSize, blockSize>>>(d_N_mobil, d_Tau, 1);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_Tau, d_Tau, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("Tau[%d]: %d\n", i, h_Tau[i]);
    }

    // 释放内存
    free(h_N_mobil);
    free(h_Tau);
    cudaFree(d_N_mobil);
    cudaFree(d_Tau);

    return 0;
}
