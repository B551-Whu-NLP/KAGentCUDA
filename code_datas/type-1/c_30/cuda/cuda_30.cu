#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void sum_arrays_gpu(int* a, int* b, int* c, int size) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    // 设置数组大小
    int arraySize = 1000;

    // 分配主机端内存
    int* h_a = (int*)malloc(arraySize * sizeof(int));
    int* h_b = (int*)malloc(arraySize * sizeof(int));
    int* h_c = (int*)malloc(arraySize * sizeof(int));

    // 初始化数据
    for (int i = 0; i < arraySize; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // 分配设备端内存
    int* d_a;
    int* d_b;
    int* d_c;
    cudaMalloc((void**)&d_a, arraySize * sizeof(int));
    cudaMalloc((void**)&d_b, arraySize * sizeof(int));
    cudaMalloc((void**)&d_c, arraySize * sizeof(int));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_a, h_a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    sum_arrays_gpu<<<gridSize, blockSize>>>(d_a, d_b, d_c, arraySize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_c, d_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_c[i]);
    }

    // 释放内存
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
