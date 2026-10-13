#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void initWith(float num, float* a, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) {
        a[i] = num;
    }
}

int main() {
    // 设置数组大小
    int arraySize = 1000;

    // 设置初始值
    float initialValue = 3.0;

    // 分配主机端内存
    float* h_a = (float*)malloc(arraySize * sizeof(float));

    // 分配设备端内存
    float* d_a;
    cudaMalloc((void**)&d_a, arraySize * sizeof(float));

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    initWith<<<gridSize, blockSize>>>(initialValue, d_a, arraySize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_a, d_a, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_a[i]);
    }

    // 释放内存
    free(h_a);
    cudaFree(d_a);

    return 0;
}
