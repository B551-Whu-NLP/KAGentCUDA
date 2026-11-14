/**
 * bicg.cu: 这是一个 CUDA 版本的 PolyBench bicg 算法实现。
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
#include "bicg.h"

// 错误阈值
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5
#define GPU_DEVICE 0

/* 数组初始化 (保持与串行代码逻辑一致) */
static
void init_array (int m, int n,
                 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
                 DATA_TYPE POLYBENCH_1D(r,N,n),
                 DATA_TYPE POLYBENCH_1D(p,M,m))
{
    int i, j;

    for (i = 0; i < m; i++)
        p[i] = (DATA_TYPE)(i % m) / m;
    for (i = 0; i < n; i++) {
        r[i] = (DATA_TYPE)(i % n) / n;
        for (j = 0; j < m; j++)
            A[i][j] = (DATA_TYPE) (i*(j+1) % n)/n;
    }
}

static
void print_array_to_file(FILE *target, int m, int n,
                         DATA_TYPE POLYBENCH_1D(s, M, m),
                         DATA_TYPE POLYBENCH_1D(q, N, n))
{
    int i;
    for (i = 0; i < m; i++)
    {
        fprintf(target, DATA_PRINTF_MODIFIER, s[i]);
        if ((i + 1) % 20 == 0)
            fprintf(target, "\n");
    }
    for (i = 0; i < n; i++)
    {
        fprintf(target, DATA_PRINTF_MODIFIER, q[i]);
        if ((i + 1) % 20 == 0)
            fprintf(target, "\n");
    }
}

/* 打印输出结果 (用于校验) */
static
void print_array(int m, int n,
                 DATA_TYPE POLYBENCH_1D(s,M,m),
                 DATA_TYPE POLYBENCH_1D(q,N,n))
{
    int i;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("s");
    for (i = 0; i < m; i++) {
        if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
        fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, s[i]);
    }
    POLYBENCH_DUMP_END("s");
    POLYBENCH_DUMP_BEGIN("q");
    for (i = 0; i < n; i++) {
        if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
        fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, q[i]);
    }
    POLYBENCH_DUMP_END("q");
    POLYBENCH_DUMP_FINISH;
}

/* GPU Kernel 1: s = r^T * A (计算结果向量 s 的每个元素) */
__global__ void bicg_kernel1(int m, int n, DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < m)
    {
        s[j] = SCALAR_VAL(0.0);
        int i;
        for(i = 0; i < n; i++)
        {
            s[j] += r[i] * A[i * m + j];
        }
    }   
}

/* GPU Kernel 2: q = A * p (计算结果向量 q 的每个元素) */
__global__ void bicg_kernel2(int m, int n, DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *q)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n)
    {
        q[i] = SCALAR_VAL(0.0);
        int j;
        for(j = 0; j < m; j++)
        {
            q[i] += A[i * m + j] * p[j];
        }
    }
}

/* CUDA 管理函数 */
void bicgCuda(int m, int n, 
              DATA_TYPE POLYBENCH_2D(A,N,M,n,m), 
              DATA_TYPE POLYBENCH_1D(s,M,m), 
              DATA_TYPE POLYBENCH_1D(q,N,n), 
              DATA_TYPE POLYBENCH_1D(p,M,m), 
              DATA_TYPE POLYBENCH_1D(r,N,n))
{
    DATA_TYPE *A_gpu;
    DATA_TYPE *q_gpu;
    DATA_TYPE *p_gpu;
    DATA_TYPE *r_gpu;
    DATA_TYPE *s_gpu;

    // 分配显存
    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * N * M);
    cudaMalloc((void **)&r_gpu, sizeof(DATA_TYPE) * N);
    cudaMalloc((void **)&s_gpu, sizeof(DATA_TYPE) * M);
    cudaMalloc((void **)&p_gpu, sizeof(DATA_TYPE) * M);
    cudaMalloc((void **)&q_gpu, sizeof(DATA_TYPE) * N);

    // 数据传输 HtoD
    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * N * M, cudaMemcpyHostToDevice);
    cudaMemcpy(r_gpu, r, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(p_gpu, p, sizeof(DATA_TYPE) * M, cudaMemcpyHostToDevice);

    // 线程配置
    dim3 block(256);
    dim3 grid1((m + block.x - 1) / block.x);
    dim3 grid2((n + block.x - 1) / block.x);

    /* 启动计时 */
    polybench_start_instruments;

    // 执行内核
    bicg_kernel1<<< grid1, block >>>(m, n, A_gpu, r_gpu, s_gpu);
    bicg_kernel2<<< grid2, block >>>(m, n, A_gpu, p_gpu, q_gpu);
    
    cudaDeviceSynchronize();

    /* 停止计时 */
    polybench_stop_instruments;
    polybench_print_instruments;
    
    // 数据传输 DtoH
    cudaMemcpy(s, s_gpu, sizeof(DATA_TYPE) * M, cudaMemcpyDeviceToHost);
    cudaMemcpy(q, q_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);

    // 释放资源
    cudaFree(A_gpu);
    cudaFree(r_gpu);
    cudaFree(s_gpu);
    cudaFree(p_gpu);
    cudaFree(q_gpu);
}

int main(int argc, char** argv)
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }

    /* 获取问题规模 */
    int n = N;
    int m = M;

    /* 变量声明 */
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, M, n, m);
    POLYBENCH_1D_ARRAY_DECL(s, DATA_TYPE, M, m);
    POLYBENCH_1D_ARRAY_DECL(q, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, M, m);
    POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);

    /* 初始化 */
    init_array(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(p));

    /* GPU 执行 */
    cudaSetDevice(GPU_DEVICE);
    bicgCuda(m, n, 
             POLYBENCH_ARRAY(A), 
             POLYBENCH_ARRAY(s), 
             POLYBENCH_ARRAY(q), 
             POLYBENCH_ARRAY(p), 
             POLYBENCH_ARRAY(r));

    /* 防止死代码消除并输出 */
    if (verbose)
    {
        FILE *output_file = fopen("c_bicg_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, m, n, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(q)));
            fclose(output_file);
        }
    }

    /* 清理内存 */
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(s);
    POLYBENCH_FREE_ARRAY(q);
    POLYBENCH_FREE_ARRAY(p);
    POLYBENCH_FREE_ARRAY(r);

    return 0;
}
