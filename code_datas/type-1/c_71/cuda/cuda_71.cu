#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void kmeans_average(int* means, int* counts) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (counts[index] == 0)
        means[index] = 0;
    else
        means[index] /= counts[index];
}

int main() {
    const int K = 10;  // 请根据实际情况修改聚类中心的数量
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (K + threadsPerBlock - 1) / threadsPerBlock;

    // 分配主机端内存
    int* h_means = (int*)malloc(K * sizeof(int));
    int* h_counts = (int*)malloc(K * sizeof(int));

    // 初始化数组数据（这里只是示例，你需要根据实际情况初始化）
    for (int i = 0; i < K; ++i) {
        h_means[i] = i * 10;
        h_counts[i] = i + 1;
    }

    // 分配设备端内存
    int* d_means;
    int* d_counts;
    cudaMalloc((void**)&d_means, K * sizeof(int));
    cudaMalloc((void**)&d_counts, K * sizeof(int));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_means, h_means, K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, h_counts, K * sizeof(int), cudaMemcpyHostToDevice);

    // 启动内核
    kmeans_average<<<blocksPerGrid, threadsPerBlock>>>(d_means, d_counts);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_means, d_means, K * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Means after averaging:\n");
    for (int i = 0; i < K; ++i) {
        printf("%d ", h_means[i]);
    }

    // 释放内存
    free(h_means);
    free(h_counts);
    cudaFree(d_means);
    cudaFree(d_counts);

    return 0;
}
