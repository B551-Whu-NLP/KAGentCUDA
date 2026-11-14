#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void test(float* input, const int dims) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dims) {
        return;
    }

    if (tid == 0) {
        input[tid] = 0;
    }
}

int main() {
    // 设置数组大小
    int arraySize = 1000;

    // 分配主机端内存
    float* h_input = (float*)malloc(arraySize * sizeof(float));

    // 分配设备端内存
    float* d_input;
    cudaMalloc((void**)&d_input, arraySize * sizeof(float));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_input, h_input, arraySize * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    test<<<gridSize, blockSize>>>(d_input, arraySize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_input, d_input, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_input[i]);
    }

    // 释放内存
    free(h_input);
    cudaFree(d_input);

    return 0;
}
