/**
 * gemm.cu: 这是一个 CUDA 版本的 PolyBench gemm 算法实现。
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
#include "gemm.h"

/* 定义 CUDA 线程块大小 */
#ifndef DIM_THREAD_BLOCK_X
# define DIM_THREAD_BLOCK_X 32
#endif
#ifndef DIM_THREAD_BLOCK_Y
# define DIM_THREAD_BLOCK_Y 8
#endif

#define GPU_DEVICE 0

/* 数组初始化 (保持与串行 C 代码逻辑一致) */
static
void init_array(int ni, int nj, int nk,
               DATA_TYPE *alpha,
               DATA_TYPE *beta,
               DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
               DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
               DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % nk) / nk;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+2) % nj) / nj;
}

static
void print_array_to_file(FILE *target, int ni, int nj,
                         DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj))
{
    int i, j;
    for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++)
        {
            fprintf(target, DATA_PRINTF_MODIFIER, C[i][j]);
            if ((i * ni + j + 1) % 20 == 0)
                fprintf(target, "\n");
        }
}

/* 打印输出结果 (用于校验或防止DCE) */
static
void print_array(int ni, int nj,
                 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j;
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}

/* GPU Kernel: C = alpha * A * B + beta * C */
__global__ void gemm_kernel(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ni && j < nj)
    {
        C[i * nj + j] *= beta;
        DATA_TYPE val = SCALAR_VAL(0.0);
        for (int k = 0; k < nk; ++k)
        {
            val += alpha * A[i * nk + k] * B[k * nj + j];
        }
        C[i * nj + j] += val;
    }
}

/* CUDA 管理函数 */
void gemmCuda(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, 
              DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), 
              DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), 
              DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
    DATA_TYPE *A_gpu;
    DATA_TYPE *B_gpu;
    DATA_TYPE *C_gpu;

    // 分配显存
    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
    cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
    cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ);
    
    // 数据传输 HtoD
    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
    
    // 设置执行配置 (映射到输出矩阵 C 的 i, j)
    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((nj + block.x - 1) / block.x, (ni + block.y - 1) / block.y);

    /* 启动计时器 */
    polybench_start_instruments;

    gemm_kernel<<< grid, block >>>(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu);
    cudaDeviceSynchronize();

    /* 停止计时器 */
    polybench_stop_instruments;
    polybench_print_instruments;

    // 数据传输 DtoH
    cudaMemcpy(C, C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);
    
    // 释放资源
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
}

int main(int argc, char *argv[])
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }

    /* 获取问题规模 */
    int ni = NI;
    int nj = NJ;
    int nk = NK;

    /* 变量声明 */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
    POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NI, NJ, ni, nj);

    /* 初始化 */
    init_array(ni, nj, nk, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));
    
    /* 设置设备 */
    cudaSetDevice(GPU_DEVICE);
    
    /* 执行 GPU 计算 */
    gemmCuda(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));

    /* 防止死代码消除并输出结果 */
    if (verbose)
    {
        FILE *output_file = fopen("c_gemm_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, ni, nj, POLYBENCH_ARRAY(C)));
            fclose(output_file);
        }
    }

    /* 清理内存 */
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(C);

    return 0;
}
