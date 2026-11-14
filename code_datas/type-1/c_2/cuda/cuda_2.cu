#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void get_ev(double* old_arr, double* new_arr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    new_arr[tid] = old_arr[tid];
}

int main() {
    // 设置数组大小
    int numElements = 1000;

    // 分配主机端内存
    double* h_old_arr = (double*)malloc(numElements * sizeof(double));
    double* h_new_arr = (double*)malloc(numElements * sizeof(double));

    // 初始化数据
    for (int i = 0; i < numElements; ++i) {
        h_old_arr[i] = static_cast<double>(i);
    }

    // 分配设备端内存
    double* d_old_arr;
    double* d_new_arr;
    cudaMalloc((void**)&d_old_arr, numElements * sizeof(double));
    cudaMalloc((void**)&d_new_arr, numElements * sizeof(double));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_old_arr, h_old_arr, numElements * sizeof(double), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;

    // 启动内核
    get_ev<<<gridSize, blockSize>>>(d_old_arr, d_new_arr);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_new_arr, d_new_arr, numElements * sizeof(double), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_new_arr[i]);
    }

    // 释放内存
    free(h_old_arr);
    free(h_new_arr);
    cudaFree(d_old_arr);
    cudaFree(d_new_arr);

    return 0;
}
