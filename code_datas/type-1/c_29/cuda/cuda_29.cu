#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void add_arrays(int n, float* x, float* y, float* z) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        z[i] = x[i] + y[i];
    }
}

int main() {
    // 设置数组大小
    int arraySize = 1000;

    // 分配主机端内存
    float* h_x = (float*)malloc(arraySize * sizeof(float));
    float* h_y = (float*)malloc(arraySize * sizeof(float));
    float* h_z = (float*)malloc(arraySize * sizeof(float));

    // 初始化数据
    for (int i = 0; i < arraySize; ++i) {
        h_x[i] = static_cast<float>(i);
        h_y[i] = static_cast<float>(2 * i);
    }

    // 分配设备端内存
    float* d_x;
    float* d_y;
    float* d_z;
    cudaMalloc((void**)&d_x, arraySize * sizeof(float));
    cudaMalloc((void**)&d_y, arraySize * sizeof(float));
    cudaMalloc((void**)&d_z, arraySize * sizeof(float));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_x, h_x, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    add_arrays<<<gridSize, blockSize>>>(arraySize, d_x, d_y, d_z);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_z, d_z, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_z[i]);
    }

    // 释放内存
    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}
