#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void scale_dev(float* array, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        array[idx] *= scale;
    }
}

int main() {
    // 设置数组大小
    int arraySize = 1000;

    // 设置缩放因子
    float scale = 1.5f;

    // 分配主机端内存
    float* h_array = (float*)malloc(arraySize * sizeof(float));

    // 初始化数据
    for (int i = 0; i < arraySize; ++i) {
        h_array[i] = static_cast<float>(i);
    }

    // 分配设备端内存
    float* d_array;
    cudaMalloc((void**)&d_array, arraySize * sizeof(float));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_array, h_array, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    scale_dev<<<gridSize, blockSize>>>(d_array, scale, arraySize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_array, d_array, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_array[i]);
    }

    // 释放内存
    free(h_array);
    cudaFree(d_array);

    return 0;
}
