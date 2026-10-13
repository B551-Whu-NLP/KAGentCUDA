/**
 * gesummv.cu: 这是一个 CUDA 版本的 PolyBench gesummv 算法实现。
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
#include "gesummv.h"

/* 定义 CUDA 线程块大小 */
#ifndef DIM_THREAD_BLOCK_X
# define DIM_THREAD_BLOCK_X 256
#endif

#define GPU_DEVICE 0

/* 数组初始化 (保持与串行 C 代码逻辑一致) */
static
void init_array(int n,
                DATA_TYPE *alpha,
                DATA_TYPE *beta,
                DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
                DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
                DATA_TYPE POLYBENCH_1D(x,N,n))
{
    int i, j;

    *alpha = 1.5;
    *beta = 1.2;
    for (i = 0; i < n; i++)
    {
        x[i] = (DATA_TYPE)(i % n) / n;
        for (j = 0; j < n; j++) {
            A[i][j] = (DATA_TYPE) ((i*j+1) % n) / n;
            B[i][j] = (DATA_TYPE) ((i*j+2) % n) / n;
        }
    }
}

static
void print_array_to_file(FILE *target, int n,
                         DATA_TYPE POLYBENCH_1D(y, N, n))
{
    int i;
    for (i = 0; i < n; i++) {
        fprintf(target, DATA_PRINTF_MODIFIER, y[i]);
        if ((i + 1) % 20 == 0) fprintf(target, "\n");
    }
}

/* 打印输出结果 (用于校验或防止 DCE) */
static
void print_array(int n,
                 DATA_TYPE POLYBENCH_1D(y,N,n))
{
    int i;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("y");
    for (i = 0; i < n; i++) {
        if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
        fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
    }
    POLYBENCH_DUMP_END("y");
    POLYBENCH_DUMP_FINISH;
}

/* GPU Kernel: 计算 tmp = Ax, y = Bx, 然后 y = alpha*tmp + beta*y */
__global__ void gesummv_kernel(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* tmp, DATA_TYPE* x, DATA_TYPE* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        DATA_TYPE tmp_val = SCALAR_VAL(0.0);
        DATA_TYPE y_val = SCALAR_VAL(0.0);
        
        for(int j = 0; j < n; j++)
        {   
            tmp_val += A[i * n + j] * x[j];
            y_val += B[i * n + j] * x[j];
        }
        
        tmp[i] = tmp_val;
        y[i] = alpha * tmp_val + beta * y_val;
    }
}

/* CUDA 管理函数 */
void gesummvCuda(int n, DATA_TYPE alpha, DATA_TYPE beta, 
                 DATA_TYPE POLYBENCH_2D(A,N,N,n,n), 
                 DATA_TYPE POLYBENCH_2D(B,N,N,n,n), 
                 DATA_TYPE POLYBENCH_1D(tmp,N,n), 
                 DATA_TYPE POLYBENCH_1D(x,N,n), 
                 DATA_TYPE POLYBENCH_1D(y,N,n))
{
    DATA_TYPE *A_gpu, *B_gpu, *x_gpu, *y_gpu, *tmp_gpu;

    // 分配显存
    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * N * N);
    cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * N * N);
    cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * N);
    cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * N);
    cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * N);
    
    // 数据传输 HtoD
    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);

    // 线程配置
    dim3 block(DIM_THREAD_BLOCK_X);
    dim3 grid((n + block.x - 1) / block.x);

    /* 启动计时器 */
    polybench_start_instruments;

    gesummv_kernel<<< grid, block >>>(n, alpha, beta, A_gpu, B_gpu, tmp_gpu, x_gpu, y_gpu);
    cudaDeviceSynchronize();

    /* 停止并打印计时 */
    polybench_stop_instruments;
    polybench_print_instruments;

    // 数据传输 DtoH
    cudaMemcpy(y, y_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);

    // 释放资源
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(x_gpu);
    cudaFree(y_gpu);
    cudaFree(tmp_gpu);
}

int main(int argc, char *argv[])
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }

    /* 获取问题规模 */
    int n = N;

    /* 变量声明 */
    DATA_TYPE alpha, beta;
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);

    /* 初始化 */
    init_array(n, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(x));
    
    /* 设置设备并执行 */
    cudaSetDevice(GPU_DEVICE);
    gesummvCuda(n, alpha, beta, 
                POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), 
                POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(x), 
                POLYBENCH_ARRAY(y));

    /* 防止死代码消除并输出 */
    if (verbose)
    {
        FILE *output_file = fopen("c_gesummv_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(y)));
            fclose(output_file);
        }
    }

    /* 清理内存 */
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(tmp);
    POLYBENCH_FREE_ARRAY(x);
    POLYBENCH_FREE_ARRAY(y);

    return 0;
}
