/**
 * syrk.cu: CUDA implementation of PolyBench SYRK (Symmetric rank-k update)
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
#include "syrk.h"

/* 定义 CUDA 线程块大小 */
#ifndef DIM_THREAD_BLOCK_X
# define DIM_THREAD_BLOCK_X 32
#endif
#ifndef DIM_THREAD_BLOCK_Y
# define DIM_THREAD_BLOCK_Y 8
#endif

#define GPU_DEVICE 0

/* 数组初始化：严格对应 2016 C 串行逻辑 */
static
void init_array(int n, int m,
                DATA_TYPE *alpha,
                DATA_TYPE *beta,
                DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
                DATA_TYPE POLYBENCH_2D(A,N,M,n,m))
{
    int i, j;

    *alpha = 1.5;
    *beta = 1.2;
    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++)
            A[i][j] = (DATA_TYPE) ((i*j+1)%n) / n;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            C[i][j] = (DATA_TYPE) ((i*j+2)%m) / m;
}

static void print_array_to_file(FILE *target, int n,
                                DATA_TYPE POLYBENCH_2D(C, N, N, n, n))
{
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            fprintf(target, DATA_PRINTF_MODIFIER, C[i][j]);
            if ((i * n + j + 1) % 20 == 0) fprintf(target, "\n");
        }
}

/* DCE code: 结果打印输出，用于对比校验 */
static
void print_array(int n, DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
    int i, j;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("C");
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
            fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
        }
    POLYBENCH_DUMP_END("C");
    POLYBENCH_DUMP_FINISH;
}

/* GPU Kernel: 实现 C := alpha*A*A' + beta*C (下三角 UPLO='L') */
__global__ void syrk_kernel(int n, int m, DATA_TYPE alpha, DATA_TYPE beta, 
                            DATA_TYPE *a, DATA_TYPE *c)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    // 只处理下三角部分 j <= i
    if (i < n && j <= i)
    {
        // 先计算 beta * C[i][j]
        double tmp_c = (double)c[i * n + j] * (double)beta;
        
        // 累加 alpha * A[i][k] * A[j][k]
        for(int k = 0; k < m; k++)
        {
            tmp_c += (double)alpha * (double)a[i * m + k] * (double)a[j * m + k];
        }
        c[i * n + j] = (DATA_TYPE)tmp_c;
    }
}

void syrkCuda(int n, int m, DATA_TYPE alpha, DATA_TYPE beta, 
              DATA_TYPE POLYBENCH_2D(A, N, M, n, m), 
              DATA_TYPE POLYBENCH_2D(C, N, N, n, n)) 
{
    DATA_TYPE *A_gpu;
    DATA_TYPE *C_gpu;

    // 分配 GPU 内存
    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * N * M);
    cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * N * N);

    // 拷贝数据
    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * N * M, cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
    
    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((size_t)ceil( ((float)N) / ((float)DIM_THREAD_BLOCK_X) ), 
              (size_t)ceil( ((float)N) / ((float)DIM_THREAD_BLOCK_Y) ));
    
    /* 计时开始 */
    polybench_start_instruments;

    syrk_kernel<<<grid, block>>>(n, m, alpha, beta, A_gpu, C_gpu);
    
    cudaDeviceSynchronize();

    /* 计时结束 */
    polybench_stop_instruments;
    polybench_print_instruments;
        
    // 拷贝结果回 Host
    cudaMemcpy(C, C_gpu, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost);

    // 释放资源
    cudaFree(A_gpu);
    cudaFree(C_gpu);
}

int main(int argc, char** argv)
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }

    /* 设置问题规模 */
    int n = N;
    int m = M;

    /* 变量定义 */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, M, n, m);

    /* 初始化数组 */
    init_array(n, m, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));
    
    /* GPU 设备选择 */
    cudaSetDevice(GPU_DEVICE);
    
    /* 执行 CUDA 计算 */
    syrkCuda(n, m, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C));

    /* 防止死代码消除并输出 */
    if (verbose)
    {
        FILE *output_file = fopen("c_syrk_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(C)));
            fclose(output_file);
        }
    }

    /* 内存清理 */
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(A);

    return 0;
}
