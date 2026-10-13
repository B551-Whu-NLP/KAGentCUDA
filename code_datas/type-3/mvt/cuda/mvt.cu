/**
 * mvt.cu: CUDA implementation of PolyBench MVT (Matrix Vector Product and Transpose)
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
#include "mvt.h"

/* 定义 CUDA 线程块大小 */
#ifndef DIM_THREAD_BLOCK_X
# define DIM_THREAD_BLOCK_X 256
#endif

#define GPU_DEVICE 0

/* 数组初始化：严格对应 2016 C 串行代码逻辑 */
static
void init_array(int n,
               DATA_TYPE POLYBENCH_1D(x1,N,n),
               DATA_TYPE POLYBENCH_1D(x2,N,n),
               DATA_TYPE POLYBENCH_1D(y_1,N,n),
               DATA_TYPE POLYBENCH_1D(y_2,N,n),
               DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        x1[i] = (DATA_TYPE) (i % n) / n;
        x2[i] = (DATA_TYPE) ((i + 1) % n) / n;
        y_1[i] = (DATA_TYPE) ((i + 3) % n) / n;
        y_2[i] = (DATA_TYPE) ((i + 4) % n) / n;
        for (j = 0; j < n; j++) {
            A[i][j] = (DATA_TYPE) (i*j % n) / n;
        }
    }
}

static void print_array_to_file(FILE *target, int n,
                                DATA_TYPE POLYBENCH_1D(x1, N, n),
                                DATA_TYPE POLYBENCH_1D(x2, N, n))
{
    int i;
    for (i = 0; i < n; i++) {
        fprintf(target, DATA_PRINTF_MODIFIER, x1[i]);
        if ((i + 1) % 20 == 0) fprintf(target, "\n");
    }
    for (i = 0; i < n; i++) {
        fprintf(target, DATA_PRINTF_MODIFIER, x2[i]);
        if ((i + 1) % 20 == 0) fprintf(target, "\n");
    }
}

/* DCE code: 打印输出结果，严格对应 PolyBench 2016 规范 */
static
void print_array(int n,
                 DATA_TYPE POLYBENCH_1D(x1,N,n),
                 DATA_TYPE POLYBENCH_1D(x2,N,n))
{
    int i;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("x1");
    for (i = 0; i < n; i++) {
        if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
        fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x1[i]);
    }
    POLYBENCH_DUMP_END("x1");

    POLYBENCH_DUMP_BEGIN("x2");
    for (i = 0; i < n; i++) {
        if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
        fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x2[i]);
    }
    POLYBENCH_DUMP_END("x2");
    POLYBENCH_DUMP_FINISH;
}

/* GPU Kernel 1: 计算 x1[i] = x1[i] + A[i][j] * y_1[j] */
__global__ void mvt_kernel1(int n, DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *y_1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < _PB_N)
    {
        // 为了提高数值健壮性，使用 double 进行中间累加
        double tmp = (double)x1[i];
        for(int j=0; j < _PB_N; j++)
        {
            tmp += (double)a[i * N + j] * (double)y_1[j];
        }
        x1[i] = (DATA_TYPE)tmp;
    }
}

/* GPU Kernel 2: 计算 x2[i] = x2[i] + A[j][i] * y_2[j] (矩阵转置乘法) */
__global__ void mvt_kernel2(int n, DATA_TYPE *a, DATA_TYPE *x2, DATA_TYPE *y_2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < _PB_N)
    {
        double tmp = (double)x2[i];
        for(int j=0; j < _PB_N; j++)
        {
            // 注意这里索引是 a[j * N + i]，实现矩阵转置效果
            tmp += (double)a[j * N + i] * (double)y_2[j]; 
        }
        x2[i] = (DATA_TYPE)tmp;
    }
}

void mvtCuda(int n, DATA_TYPE POLYBENCH_2D(a, N, N, n, n), 
             DATA_TYPE POLYBENCH_1D(x1, N, n), DATA_TYPE POLYBENCH_1D(x2, N, n), 
             DATA_TYPE POLYBENCH_1D(y_1, N, n), DATA_TYPE POLYBENCH_1D(y_2, N, n))
{
    DATA_TYPE* a_gpu;
    DATA_TYPE* x1_gpu;
    DATA_TYPE* x2_gpu;
    DATA_TYPE* y_1_gpu;
    DATA_TYPE* y_2_gpu;

    // 分配 GPU 内存
    cudaMalloc((void **)&a_gpu, sizeof(DATA_TYPE) * N * N);
    cudaMalloc((void **)&x1_gpu, sizeof(DATA_TYPE) * N);
    cudaMalloc((void **)&x2_gpu, sizeof(DATA_TYPE) * N);
    cudaMalloc((void **)&y_1_gpu, sizeof(DATA_TYPE) * N);
    cudaMalloc((void **)&y_2_gpu, sizeof(DATA_TYPE) * N);

    // 拷贝数据到 GPU
    cudaMemcpy(a_gpu, a, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(x1_gpu, x1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(x2_gpu, x2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(y_1_gpu, y_1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(y_2_gpu, y_2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
    
    dim3 block(DIM_THREAD_BLOCK_X);
    dim3 grid((size_t)ceil((float)N / ((float)DIM_THREAD_BLOCK_X)));

    /* 启动计时 */
    polybench_start_instruments;
    
    // 执行 Kernel 1
    mvt_kernel1<<<grid, block>>>(n, a_gpu, x1_gpu, y_1_gpu);
    // 执行 Kernel 2
    mvt_kernel2<<<grid, block>>>(n, a_gpu, x2_gpu, y_2_gpu);
    
    cudaDeviceSynchronize();

    /* 停止并打印计时 */
    polybench_stop_instruments;
    polybench_print_instruments;
    
    // 拷贝回结果
    cudaMemcpy(x1, x1_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(x2, x2_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost); 
    
    // 释放资源
    cudaFree(a_gpu);
    cudaFree(x1_gpu);
    cudaFree(x2_gpu);
    cudaFree(y_1_gpu);
    cudaFree(y_2_gpu);
}

int main(int argc, char *argv[])
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }

    int n = N;

    /* 变量声明/分配 */
    POLYBENCH_2D_ARRAY_DECL(a, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(x1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y_1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y_2, DATA_TYPE, N, n);

    /* 初始化数组 */
    init_array(n, POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2), 
               POLYBENCH_ARRAY(y_1), POLYBENCH_ARRAY(y_2), 
               POLYBENCH_ARRAY(a));
    
    /* CUDA 初始化 */
    cudaSetDevice(GPU_DEVICE);

    /* 执行计算 */
    mvtCuda(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2), 
            POLYBENCH_ARRAY(y_1), POLYBENCH_ARRAY(y_2));

    /* 防止死代码消除 (DCE) 并输出结果 */
    if (verbose)
    {
        FILE *output_file = fopen("c_mvt_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2)));
            fclose(output_file);
        }
    }

    /* 释放内存 */
    POLYBENCH_FREE_ARRAY(a);
    POLYBENCH_FREE_ARRAY(x1);
    POLYBENCH_FREE_ARRAY(x2);
    POLYBENCH_FREE_ARRAY(y_1);
    POLYBENCH_FREE_ARRAY(y_2);

    return 0;
}
