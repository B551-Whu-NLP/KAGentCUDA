#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void set_sorting_offset(const int nrows, const int ncols, int* offsets) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid <= ncols) {
        offsets[tid] = tid * nrows;
    }
}

int main() {
    // 设置行数和列数
    int nrows = 100;
    int ncols = 10;

    // 分配主机端内存
    int* h_offsets = (int*)malloc((ncols + 1) * sizeof(int));

    // 分配设备端内存
    int* d_offsets;
    cudaMalloc((void**)&d_offsets, (ncols + 1) * sizeof(int));

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (ncols + blockSize - 1) / blockSize;

    // 启动内核
    set_sorting_offset<<<gridSize, blockSize>>>(nrows, ncols, d_offsets);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_offsets, d_offsets, (ncols + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (int i = 0; i < ncols + 1; ++i) {
        printf("%d ", h_offsets[i]);
    }

    // 释放内存
    free(h_offsets);
    cudaFree(d_offsets);

    return 0;
}
