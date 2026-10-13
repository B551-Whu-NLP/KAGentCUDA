#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void doubleArrayScalarDivideKernel(double* d_in, int* d_out, int length, double scalar) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < length) {
        d_out[tid] = static_cast<int>(d_in[tid] / scalar);
    }
}

int main() {
    // 设置数组大小
    const int arraySize = 1000;

    // 分配主机端内存
    double* h_d_in = (double*)malloc(arraySize * sizeof(double));
    int* h_d_out = (int*)malloc(arraySize * sizeof(int));
    const double scalar = 2.0;  // 设置一个常量值

    // 初始化数据
    for (int i = 0; i < arraySize; ++i) {
        h_d_in[i] = static_cast<double>(i);
    }

    // 分配设备端内存
    double* d_d_in;
    int* d_d_out;
    cudaMalloc((void**)&d_d_in, arraySize * sizeof(double));
    cudaMalloc((void**)&d_d_out, arraySize * sizeof(int));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_d_in, h_d_in, arraySize * sizeof(double), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    doubleArrayScalarDivideKernel<<<gridSize, blockSize>>>(d_d_in, d_d_out, arraySize, scalar);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_d_out, d_d_out, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_d_out[i]);
    }

    // 释放内存
    free(h_d_in);
    free(h_d_out);
    cudaFree(d_d_in);
    cudaFree(d_d_out);

    return 0;
}
