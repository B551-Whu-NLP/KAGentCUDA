#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void iKernel(float* A, float* B, float* C, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // 设置数组大小
    const int arraySize = 1000;

    // 分配主机端内存
    float* h_A = (float*)malloc(arraySize * sizeof(float));
    float* h_B = (float*)malloc(arraySize * sizeof(float));
    float* h_C = (float*)malloc(arraySize * sizeof(float));

    // 初始化数据
    for (int i = 0; i < arraySize; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // 分配设备端内存
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**)&d_A, arraySize * sizeof(float));
    cudaMalloc((void**)&d_B, arraySize * sizeof(float));
    cudaMalloc((void**)&d_C, arraySize * sizeof(float));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_A, h_A, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    iKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, arraySize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_C, d_C, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_C[i]);
    }

    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
