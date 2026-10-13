#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fill_kernel(int N, float ALPHA, float* X, int INCX) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N)
        X[i * INCX] = ALPHA;
}

int main() {
    // 设置数组大小
    int arraySize = 1000;

    // 设置填充值和步长
    float ALPHA = 2.0;
    int INCX = 2;

    // 分配主机端内存
    float* h_X = (float*)malloc(arraySize * sizeof(float));

    // 分配设备端内存
    float* d_X;
    cudaMalloc((void**)&d_X, arraySize * sizeof(float));

    // 设置线程块和网格大小
    dim3 blockSize(256, 1, 1);
    dim3 gridSize((arraySize + blockSize.x - 1) / blockSize.x, 1, 1);

    // 启动内核
    fill_kernel<<<gridSize, blockSize>>>(arraySize, ALPHA, d_X, INCX);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_X, d_X, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_X[i]);
    }

    // 释放内存
    free(h_X);
    cudaFree(d_X);

    return 0;
}
