#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void compute_array_square(float* array, float* outArray, int size) {
    int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;

    for (int i = 0; i < size; i += num_threads) {
        int index = i + thread_index;

        if (index < size) {
            outArray[index] = array[index] * array[index];
        }
    }
}

int main() {
    // 定义数组大小
    const int size = 1000;

    // 分配主机端内存
    float* h_array = (float*)malloc(size * sizeof(float));
    float* h_outArray = (float*)malloc(size * sizeof(float));

    // 初始化数组数据
    for (int i = 0; i < size; ++i) {
        h_array[i] = static_cast<float>(i);
    }

    // 分配设备端内存
    float* d_array;
    float* d_outArray;
    cudaMalloc((void**)&d_array, size * sizeof(float));
    cudaMalloc((void**)&d_outArray, size * sizeof(float));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_array, h_array, size * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(256);
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x, 1);

    // 启动内核
    compute_array_square<<<gridSize, blockSize>>>(d_array, d_outArray, size);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_outArray, d_outArray, size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        printf("outArray[%d]: %f\n", i, h_outArray[i]);
    }

    // 释放内存
    free(h_array);
    free(h_outArray);
    cudaFree(d_array);
    cudaFree(d_outArray);

    return 0;
}
