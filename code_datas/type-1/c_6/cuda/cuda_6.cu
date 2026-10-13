#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void allAddInplaceKernel(double* arr, double alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        arr[i] += alpha;
    }
}

int main() {
    // 设置数组大小
    int arraySize = 1000;

    // 设置增量
    double alpha = 5.0;

    // 分配主机端内存
    double* h_arr = (double*)malloc(arraySize * sizeof(double));

    // 初始化数据
    for (int i = 0; i < arraySize; ++i) {
        h_arr[i] = static_cast<double>(i);
    }

    // 分配设备端内存
    double* d_arr;
    cudaMalloc((void**)&d_arr, arraySize * sizeof(double));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_arr, h_arr, arraySize * sizeof(double), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    allAddInplaceKernel<<<gridSize, blockSize>>>(d_arr, alpha, arraySize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_arr, d_arr, arraySize * sizeof(double), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_arr[i]);
    }

    // 释放内存
    free(h_arr);
    cudaFree(d_arr);

    return 0;
}
