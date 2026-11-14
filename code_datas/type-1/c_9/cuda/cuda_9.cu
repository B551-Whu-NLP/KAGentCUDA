#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void VectorAdd(float* arrayA, float* arrayB, float* output) {
    int idx = threadIdx.x;
    output[idx] = arrayA[idx] + arrayB[idx];
}

int main() {
    // 设置数组大小
    int arraySize = 1000;

    // 分配主机端内存
    float* h_arrayA = (float*)malloc(arraySize * sizeof(float));
    float* h_arrayB = (float*)malloc(arraySize * sizeof(float));
    float* h_output = (float*)malloc(arraySize * sizeof(float));

    // 初始化数据
    for (int i = 0; i < arraySize; ++i) {
        h_arrayA[i] = static_cast<float>(i);
        h_arrayB[i] = static_cast<float>(2 * i);
    }

    // 分配设备端内存
    float* d_arrayA;
    float* d_arrayB;
    float* d_output;
    cudaMalloc((void**)&d_arrayA, arraySize * sizeof(float));
    cudaMalloc((void**)&d_arrayB, arraySize * sizeof(float));
    cudaMalloc((void**)&d_output, arraySize * sizeof(float));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_arrayA, h_arrayA, arraySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arrayB, h_arrayB, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    VectorAdd<<<gridSize, blockSize>>>(d_arrayA, d_arrayB, d_output);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_output, d_output, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_output[i]);
    }

    // 释放内存
    free(h_arrayA);
    free(h_arrayB);
    free(h_output);
    cudaFree(d_arrayA);
    cudaFree(d_arrayB);
    cudaFree(d_output);

    return 0;
}
