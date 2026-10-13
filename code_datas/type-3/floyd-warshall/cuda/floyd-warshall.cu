/**
 * floyd_warshall.cu: CUDA implementation of PolyBench Floyd-Warshall
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <string.h>

/* 包含 PolyBench 基础头文件 */
#define POLYBENCH_TIME 1
#include <polybench.h>
#include "floyd-warshall.h"

#ifndef DIM_THREAD_BLOCK_X
# define DIM_THREAD_BLOCK_X 32
#endif
#ifndef DIM_THREAD_BLOCK_Y
# define DIM_THREAD_BLOCK_Y 8
#endif

#define GPU_DEVICE 0

/* 数组初始化：严格对应 2016 C 串行逻辑 */
static void init_array (int n, DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            path[i][j] = (DATA_TYPE) (i * j % 7 + 1);
            if ((i + j) % 13 == 0 || (i + j) % 7 == 0 || (i + j) % 11 == 0)
                path[i][j] = (DATA_TYPE) 999;
        }
}

static void print_array_to_file(FILE *target, int n,
                                DATA_TYPE POLYBENCH_2D(path, N, N, n, n))
{
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            fprintf(target, DATA_PRINTF_MODIFIER, path[i][j]);
            if ((i * n + j + 1) % 20 == 0)
                fprintf(target, "\n");
        }
}

/* 结果打印：对应 PolyBench 2016 DUMP 规范 */
static void print_array(int n, DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
    int i, j;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("path");
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
            fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, path[i][j]);
        }
    POLYBENCH_DUMP_END("path");
    POLYBENCH_DUMP_FINISH;
}

/* CUDA Kernel: 每一轮 k 更新整个路径矩阵 */
__global__ void floyd_warshall_kernel(int n, DATA_TYPE *path, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n)
    {
        DATA_TYPE cur_dist = path[i * n + j];
        DATA_TYPE new_dist = path[i * n + k] + path[k * n + j];
        
        // 相当于 path[i][j] = min(path[i][j], path[i][k] + path[k][j])
        path[i * n + j] = (cur_dist < new_dist) ? cur_dist : new_dist;
    }
}

void floydWarshallCuda(int n, DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
    DATA_TYPE* path_gpu;
    
    // 分配显存并拷贝数据
    cudaMalloc(&path_gpu, n * n * sizeof(DATA_TYPE));
    cudaMemcpy(path_gpu, path, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    /* 启动计时 */
    polybench_start_instruments;

    // k 循环必须在 CPU 端串行发射，因为每一步 k 依赖于前一步的结果
    for (int k = 0; k < n; k++)
    {
        floyd_warshall_kernel<<<grid, block>>>(n, path_gpu, k);
        // Kernel 发射后不强制 cudaDeviceSynchronize，因为默认流中内核发射是保序的
        // 但对于这种算法，内核间的隐式同步（或流同步）至关重要
    }
    
    cudaDeviceSynchronize();

    /* 停止计时 */
    polybench_stop_instruments;
    polybench_print_instruments;

    // 拷贝回结果
    cudaMemcpy(path, path_gpu, n * n * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaFree(path_gpu);
}

int main(int argc, char** argv)
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }

    int n = N;

    /* 变量声明 */
    POLYBENCH_2D_ARRAY_DECL(path, DATA_TYPE, N, N, n, n);

    /* 初始化 */
    init_array(n, POLYBENCH_ARRAY(path));

    /* 执行 CUDA 版本 */
    cudaSetDevice(GPU_DEVICE);
    floydWarshallCuda(n, POLYBENCH_ARRAY(path));

    /* 防止死代码消除并 DUMP 结果 */
    if (verbose)
    {
        FILE *output_file = fopen("c_floyd_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(path)));
            fclose(output_file);
        }
    }

    /* 清理内存 */
    POLYBENCH_FREE_ARRAY(path);

    return 0;
}
