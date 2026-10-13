#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void mul_Scalar_matrix(float* a, float value, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] * value;
    }
}

int main() {
    // 设置数组大小
    int arraySize = 1000;

    // 设置缩放因子
    float value = 2.0;

    // 分配主机端内存
    float* h_a = (float*)malloc(arraySize * sizeof(float));
    float* h_c = (float*)malloc(arraySize * sizeof(float));

    // 初始化数据
    for (int i = 0; i < arraySize; ++i) {
        h_a[i] = static_cast<float>(i);
    }

    // 分配设备端内存
    float* d_a;
    float* d_c;
    cudaMalloc((void**)&d_a, arraySize * sizeof(float));
    cudaMalloc((void**)&d_c, arraySize * sizeof(float));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_a, h_a, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    mul_Scalar_matrix<<<gridSize, blockSize>>>(d_a, value, d_c, arraySize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_c, d_c, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_c[i]);
    }

    // 释放内存
    free(h_a);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_c);

    return 0;
}
