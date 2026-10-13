#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void cudaAddCorrAndCorrection(float* L, float* r, int N) {
    int u = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (u >= N)
        return;
    
    L[u] -= r[u];
}

int main() {
    // 设置数组大小
    int arraySize = 1000;

    // 分配主机端内存
    float* h_L = (float*)malloc(arraySize * sizeof(float));
    float* h_r = (float*)malloc(arraySize * sizeof(float));

    // 初始化数据
    for (int i = 0; i < arraySize; ++i) {
        h_L[i] = static_cast<float>(i);
        h_r[i] = static_cast<float>(2 * i);
    }

    // 分配设备端内存
    float* d_L;
    float* d_r;
    cudaMalloc((void**)&d_L, arraySize * sizeof(float));
    cudaMalloc((void**)&d_r, arraySize * sizeof(float));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_L, h_L, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, h_r, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    cudaAddCorrAndCorrection<<<gridSize, blockSize>>>(d_L, d_r, arraySize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_L, d_L, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_L[i]);
    }

    // 释放内存
    free(h_L);
    free(h_r);
    cudaFree(d_L);
    cudaFree(d_r);

    return 0;
}
