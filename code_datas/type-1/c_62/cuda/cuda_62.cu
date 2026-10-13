#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void Blending_Kernel(unsigned char* aR1, unsigned char* aR2, unsigned char* aRS, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        aRS[index] = 0.5 * aR1[index] + 0.5 * aR2[index];
    }
}

int main() {
    // 定义数组大小
    const int size = 1000;

    // 分配主机端内存
    unsigned char* h_aR1 = (unsigned char*)malloc(size * sizeof(unsigned char));
    unsigned char* h_aR2 = (unsigned char*)malloc(size * sizeof(unsigned char));
    unsigned char* h_aRS = (unsigned char*)malloc(size * sizeof(unsigned char));

    // 初始化数组数据
    for (int i = 0; i < size; ++i) {
        h_aR1[i] = static_cast<unsigned char>(i * 2); // Just an example, you can initialize it according to your needs
        h_aR2[i] = static_cast<unsigned char>(i * 3); // Just an example, you can initialize it according to your needs
    }

    // 分配设备端内存
    unsigned char* d_aR1;
    unsigned char* d_aR2;
    unsigned char* d_aRS;
    cudaMalloc((void**)&d_aR1, size * sizeof(unsigned char));
    cudaMalloc((void**)&d_aR2, size * sizeof(unsigned char));
    cudaMalloc((void**)&d_aRS, size * sizeof(unsigned char));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_aR1, h_aR1, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_aR2, h_aR2, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    Blending_Kernel<<<gridSize, blockSize>>>(d_aR1, d_aR2, d_aRS, size);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_aRS, d_aRS, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("h_aRS[%d]: %u\n", i, h_aRS[i]);
    }

    // 释放内存
    free(h_aR1);
    free(h_aR2);
    free(h_aRS);
    cudaFree(d_aR1);
    cudaFree(d_aR2);
    cudaFree(d_aRS);

    return 0;
}
