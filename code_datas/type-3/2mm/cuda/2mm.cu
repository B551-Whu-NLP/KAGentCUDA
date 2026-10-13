/**
 * 2mm.cu: 这是一个 CUDA 版本的 PolyBench 2mm 算法实现。
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>

/* 包含 PolyBench 基础头文件 */
#define POLYBENCH_TIME 1
#include <polybench.h>
#include "2mm.h"

/* 定义 CUDA 线程块大小 */
#ifndef DIM_THREAD_BLOCK_X
# define DIM_THREAD_BLOCK_X 32
#endif
#ifndef DIM_THREAD_BLOCK_Y
# define DIM_THREAD_BLOCK_Y 8
#endif

/* 初始化数组 (保持与原始 C 代码逻辑一致) */
static
void init_array(int ni, int nj, int nk, int nl,
                DATA_TYPE *alpha,
                DATA_TYPE *beta,
                DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
                DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
                DATA_TYPE POLYBENCH_2D(C, NJ, NL, nj, nl),
                DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl))
{
    int i, j;
    *alpha = 1.5;
    *beta = 1.2;
    for (i = 0; i < ni; i++)
        for (j = 0; j < nk; j++)
            A[i][j] = (DATA_TYPE)((i * j + 1) % ni) / ni;
    for (i = 0; i < nk; i++)
        for (j = 0; j < nj; j++)
            B[i][j] = (DATA_TYPE)(i * (j + 1) % nj) / nj;
    for (i = 0; i < nj; i++)
        for (j = 0; j < nl; j++)
            C[i][j] = (DATA_TYPE)((i * (j + 3) + 1) % nl) / nl;
    for (i = 0; i < ni; i++)
        for (j = 0; j < nl; j++)
            D[i][j] = (DATA_TYPE)(i * (j + 2) % nk) / nk;
}

/* 打印结果 (用于防止死代码消除) */
static
void print_array(int ni, int nl,
                 DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl))
{
    int i, j;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("D");
    for (i = 0; i < ni; i++)
        for (j = 0; j < nl; j++) {
            if ((i * ni + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
            fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, D[i][j]);
        }
    POLYBENCH_DUMP_END("D");
    POLYBENCH_DUMP_FINISH;
}

/* GPU Kernel 1: tmp = alpha * A * B */
__global__ void mm2_kernel1(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *tmp)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ni && j < nj) {
        DATA_TYPE val = SCALAR_VAL(0.0);
        for (int k = 0; k < nk; ++k) {
            val += alpha * A[i * nk + k] * B[k * nj + j];
        }
        tmp[i * nj + j] = val;
    }
}

/* GPU Kernel 2: D = tmp * C + beta * D */
__global__ void mm2_kernel2(int ni, int nj, int nl, DATA_TYPE beta, DATA_TYPE *tmp, DATA_TYPE *C, DATA_TYPE *D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ni && j < nl) {
        D[i * nl + j] *= beta;
        for (int k = 0; k < nj; ++k) {
            D[i * nl + j] += tmp[i * nj + k] * C[k * nl + j];
        }
    }
}

static void print_array_to_file(FILE *target, int ni, int nl,
                                DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl))
{
  int i, j;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
    {
      fprintf(target, DATA_PRINTF_MODIFIER, D[i][j]);
      if ((i * ni + j + 1) % 20 == 0)
        fprintf(target, "\n");
    }
}

int main(int argc, char** argv)
{
    
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }
    int ni = NI; int nj = NJ; int nk = NK; int nl = NL;

    /* 变量声明 */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(tmp, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
    POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NL, nj, nl);
    POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NI, NL, ni, nl);

    /* 初始化 */
    init_array(ni, nj, nk, nl, &alpha, &beta, 
               POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), 
               POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));

    /* GPU 内存分配 */
    DATA_TYPE *A_gpu, *B_gpu, *C_gpu, *D_gpu, *tmp_gpu;
    cudaMalloc((void **)&A_gpu, NI * NK * sizeof(DATA_TYPE));
    cudaMalloc((void **)&B_gpu, NK * NJ * sizeof(DATA_TYPE));
    cudaMalloc((void **)&C_gpu, NJ * NL * sizeof(DATA_TYPE));
    cudaMalloc((void **)&D_gpu, NI * NL * sizeof(DATA_TYPE));
    cudaMalloc((void **)&tmp_gpu, NI * NJ * sizeof(DATA_TYPE));

    /* 数据传输 HtoD */
    cudaMemcpy(A_gpu, POLYBENCH_ARRAY(A), NI * NK * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, POLYBENCH_ARRAY(B), NK * NJ * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, POLYBENCH_ARRAY(C), NJ * NL * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(D_gpu, POLYBENCH_ARRAY(D), NI * NL * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    /* 定义网格与线程块 */
    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid1((nj + block.x - 1) / block.x, (ni + block.y - 1) / block.y);
    dim3 grid2((nl + block.x - 1) / block.x, (ni + block.y - 1) / block.y);

    /* 启动计时器 */
    polybench_start_instruments;

    /* 执行计算核心 */
    mm2_kernel1<<<grid1, block>>>(ni, nj, nk, alpha, A_gpu, B_gpu, tmp_gpu);
    cudaDeviceSynchronize();
    mm2_kernel2<<<grid2, block>>>(ni, nj, nl, beta, tmp_gpu, C_gpu, D_gpu);
    cudaDeviceSynchronize();

    /* 停止计时器 */
    polybench_stop_instruments;
    polybench_print_instruments;

    /* 数据传输 DtoH */
    cudaMemcpy(POLYBENCH_ARRAY(D), D_gpu, NI * NL * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    /* 防止 DCE 并打印 */
    if (verbose)
  {
    FILE *output_file = fopen("c_2mm_output.txt", "w");
    if (output_file != NULL)
    {
      polybench_prevent_dce(print_array_to_file(output_file, ni, nl, POLYBENCH_ARRAY(D)));
      fclose(output_file);
    }
  }

    /* 清理内存 */
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    cudaFree(D_gpu);
    cudaFree(tmp_gpu);
    POLYBENCH_FREE_ARRAY(tmp);
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(D);

    return 0;
}