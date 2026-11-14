#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void compute_new_means(float* mx, float* my, const float* sx, const float* sy, const int* c) {
    const int cluster = threadIdx.x;

    if (cluster < blockDim.x) {
        const int count = max(1, c[cluster]);
        mx[cluster] = sx[cluster] / count;
        my[cluster] = sy[cluster] / count;
    }
}

int main() {
    const int clusters = 5;  // 请根据实际情况修改簇的数量
    const int threadsPerBlock = clusters;
    const int blocksPerGrid = 1;  // 因为每个块只处理一个簇，所以块的数量为1

    // 分配主机端内存
    float* h_mx = (float*)malloc(clusters * sizeof(float));
    float* h_my = (float*)malloc(clusters * sizeof(float));
    float* h_sx = (float*)malloc(clusters * sizeof(float));
    float* h_sy = (float*)malloc(clusters * sizeof(float));
    int* h_c = (int*)malloc(clusters * sizeof(int));

    // 初始化数组数据（这里只是示例，你需要根据实际情况初始化）
    for (int i = 0; i < clusters; ++i) {
        h_sx[i] = i + 1;
        h_sy[i] = i + 1;
        h_c[i] = i + 1;
    }

    // 分配设备端内存
    float* d_mx;
    float* d_my;
    float* d_sx;
    float* d_sy;
    int* d_c;
    cudaMalloc((void**)&d_mx, clusters * sizeof(float));
    cudaMalloc((void**)&d_my, clusters * sizeof(float));
    cudaMalloc((void**)&d_sx, clusters * sizeof(float));
    cudaMalloc((void**)&d_sy, clusters * sizeof(float));
    cudaMalloc((void**)&d_c, clusters * sizeof(int));

    // 将数组数据从主机端拷贝到设备端
    cudaMemcpy(d_mx, h_mx, clusters * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_my, h_my, clusters * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sx, h_sx, clusters * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sy, h_sy, clusters * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, clusters * sizeof(int), cudaMemcpyHostToDevice);

    // 启动内核
    compute_new_means<<<blocksPerGrid, threadsPerBlock>>>(d_mx, d_my, d_sx, d_sy, d_c);

    // 将结果从设备端拷贝回主机端
    cudaMemcpy(h_mx, d_mx, clusters * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_my, d_my, clusters * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("New means:\n");
    for (int i = 0; i < clusters; ++i) {
        printf("(%f, %f)\n", h_mx[i], h_my[i]);
    }

    // 释放内存
    free(h_mx);
    free(h_my);
    free(h_sx);
    free(h_sy);
    free(h_c);
    cudaFree(d_mx);
    cudaFree(d_my);
    cudaFree(d_sx);
    cudaFree(d_sy);
    cudaFree(d_c);

    return 0;
}
