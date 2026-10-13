#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void initialArray0(int tasks, int* f3) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < tasks; i += blockDim.x * gridDim.x) {
        f3[i] = 0;
    }
}

int main() {
    // 设置任务数
    int numTasks = 1000;

    // 分配主机端内存
    int* h_f3 = (int*)malloc(numTasks * sizeof(int));

    // 分配设备端内存
    int* d_f3;
    cudaMalloc((void**)&d_f3, numTasks * sizeof(int));

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (numTasks + blockSize - 1) / blockSize;

    // 启动内核
    initialArray0<<<gridSize, blockSize>>>(numTasks, d_f3);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_f3, d_f3, numTasks * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_f3[i]);
    }

    // 释放内存
    free(h_f3);
    cudaFree(d_f3);

    return 0;
}
