#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void getCanBusData(int* canData, int size, int nthreads, int nblocks) {
    int i;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (i = idx; i < size; i += nthreads * nblocks) {
        atomicAdd(&canData[i], 1);
    }
}

int main() {
    // 设置数组大小
    int arraySize = 1000;

    // 分配主机端内存
    int* h_canData = (int*)malloc(arraySize * sizeof(int));

    // 分配设备端内存
    int* d_canData;
    cudaMalloc((void**)&d_canData, arraySize * sizeof(int));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_canData, h_canData, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    getCanBusData<<<gridSize, blockSize>>>(d_canData, arraySize, blockSize, gridSize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_canData, d_canData, arraySize * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_canData[i]);
    }

    // 释放内存
    free(h_canData);
    cudaFree(d_canData);

    return 0;
}
