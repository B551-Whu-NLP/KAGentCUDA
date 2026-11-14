#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void zeroIndices(long* vec_out, const long N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        vec_out[idx] = vec_out[idx] - vec_out[0];
    }
}

int main() {
    // 设置数组大小
    long arraySize = 1000;

    // 分配主机端内存
    long* h_vec_out = (long*)malloc(arraySize * sizeof(long));

    // 初始化数据
    for (long i = 0; i < arraySize; ++i) {
        h_vec_out[i] = static_cast<long>(i);
    }

    // 分配设备端内存
    long* d_vec_out;
    cudaMalloc((void**)&d_vec_out, arraySize * sizeof(long));

    // 将数据从主机端拷贝到设备端
    cudaMemcpy(d_vec_out, h_vec_out, arraySize * sizeof(long), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int blockSize = 256;
    int gridSize = (arraySize + blockSize - 1) / blockSize;

    // 启动内核
    zeroIndices<<<gridSize, blockSize>>>(d_vec_out, arraySize);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_vec_out, d_vec_out, arraySize * sizeof(long), cudaMemcpyDeviceToHost);

    // 打印结果（为了简化，此处不打印全部结果）
    for (long i = 0; i < 10; ++i) {
        printf("%ld ", h_vec_out[i]);
    }

    // 释放内存
    free(h_vec_out);
    cudaFree(d_vec_out);

    return 0;
}
