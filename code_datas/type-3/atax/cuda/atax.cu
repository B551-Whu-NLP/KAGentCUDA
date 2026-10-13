/**
 * atax.cu: 这是一个 CUDA 版本的 PolyBench atax 算法实现。
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>

/* 包含 PolyBench 基础头文件 */
#define POLYBENCH_TIME 1

#include <polybench.h>
#include "atax.h"

/* 定义 CUDA 线程块大小 */
#ifndef DIM_THREAD_BLOCK_X
#define DIM_THREAD_BLOCK_X 256
#endif

#define GPU_DEVICE 0

/* 数组初始化 (保持与串行代码逻辑一致) */
static void init_array(int m, int n,
                       DATA_TYPE POLYBENCH_2D(A, M, N, m, n),
                       DATA_TYPE POLYBENCH_1D(x, N, n))
{
    int i, j;
    DATA_TYPE fn = (DATA_TYPE)n;

    for (i = 0; i < n; i++)
        x[i] = 1 + (i / fn);
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            A[i][j] = (DATA_TYPE)((i + j) % n) / (5 * m);
}
static void print_array_to_file(FILE *target, int n,
                                DATA_TYPE POLYBENCH_1D(y, N, n))
{
    int i;
    for (i = 0; i < n; i++)
    {
        fprintf(target, DATA_PRINTF_MODIFIER, y[i]);
        if ((i + 1) % 20 == 0)
            fprintf(target, "\n");
    }
}

/* 打印输出结果 (用于校验或防止DCE) */
static void print_array(int n,
                        DATA_TYPE POLYBENCH_1D(y, N, n))
{
    int i;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("y");
    for (i = 0; i < n; i++)
    {
        if (i % 20 == 0)
            fprintf(POLYBENCH_DUMP_TARGET, "\n");
        fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
    }
    POLYBENCH_DUMP_END("y");
    POLYBENCH_DUMP_FINISH;
}

/* GPU Kernel 1: tmp = A * x */
__global__ void atax_kernel1(int m, int n, DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m)
    {
        DATA_TYPE val = SCALAR_VAL(0.0);
        int j;
        for (j = 0; j < n; j++)
        {
            val += A[i * n + j] * x[j];
        }
        tmp[i] = val;
    }
}

/* GPU Kernel 2: y = A^T * tmp */
__global__ void atax_kernel2(int m, int n, DATA_TYPE *A, DATA_TYPE *tmp, DATA_TYPE *y)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < n)
    {
        DATA_TYPE val = SCALAR_VAL(0.0);
        int i;
        for (i = 0; i < m; i++)
        {
            val += A[i * n + j] * tmp[i];
        }
        y[j] = val;
    }
}

/* CUDA 管理函数 */
void ataxGpu(int m, int n,
             DATA_TYPE POLYBENCH_2D(A, M, N, m, n),
             DATA_TYPE POLYBENCH_1D(x, N, n),
             DATA_TYPE POLYBENCH_1D(y, N, n),
             DATA_TYPE POLYBENCH_1D(tmp, M, m))
{
    DATA_TYPE *A_gpu;
    DATA_TYPE *x_gpu;
    DATA_TYPE *y_gpu;
    DATA_TYPE *tmp_gpu;

    // 分配显存
    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * M * N);
    cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * N);
    cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * N);
    cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * M);

    // 数据传输 HtoD
    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);

    // 设置执行配置
    dim3 block(DIM_THREAD_BLOCK_X);
    dim3 grid1((m + block.x - 1) / block.x);
    dim3 grid2((n + block.x - 1) / block.x);

    /* 启动计时 */
    polybench_start_instruments;

    // 阶段 1: tmp = A * x
    atax_kernel1<<<grid1, block>>>(m, n, A_gpu, x_gpu, tmp_gpu);
    cudaDeviceSynchronize();

    // 阶段 2: y = A' * tmp
    atax_kernel2<<<grid2, block>>>(m, n, A_gpu, tmp_gpu, y_gpu);
    cudaDeviceSynchronize();

    /* 停止计时 */
    polybench_stop_instruments;
    polybench_print_instruments;

    // 数据传输 DtoH
    cudaMemcpy(y, y_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);

    // 释放资源
    cudaFree(A_gpu);
    cudaFree(x_gpu);
    cudaFree(y_gpu);
    cudaFree(tmp_gpu);
}

int main(int argc, char **argv)
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }
    /* 获取问题规模 */
    int m = M;
    int n = N;

    /* 变量声明 */
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, N, m, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, M, m);

    /* 初始化 */
    init_array(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

    /* 设置设备 */
    cudaSetDevice(GPU_DEVICE);

    /* 执行 GPU 计算 */
    ataxGpu(m, n,
            POLYBENCH_ARRAY(A),
            POLYBENCH_ARRAY(x),
            POLYBENCH_ARRAY(y),
            POLYBENCH_ARRAY(tmp));

    /* 防止死代码消除并输出 */
    if (verbose)
    {
        FILE *output_file = fopen("c_atax_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(y)));
            fclose(output_file);
        }
    }

    /* 清理内存 */
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(x);
    POLYBENCH_FREE_ARRAY(y);
    POLYBENCH_FREE_ARRAY(tmp);

    return 0;
}
