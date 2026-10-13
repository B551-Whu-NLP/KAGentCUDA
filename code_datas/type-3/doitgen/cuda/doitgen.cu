/**
 * doitgen.cu: 这是一个 CUDA 版本的 PolyBench doitgen 算法实现。
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
#include "doitgen.h"

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
void init_array(int nr, int nq, int np,
               DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
               DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np))
{
    int i, j, k;

    for (i = 0; i < nr; i++)
        for (j = 0; j < nq; j++)
            for (k = 0; k < np; k++)
                A[i][j][k] = (DATA_TYPE) ((i*j + k) % np) / np;

    for (i = 0; i < np; i++)
        for (j = 0; j < np; j++)
            C4[i][j] = (DATA_TYPE) (i*j % np) / np;
}

static
void print_array_to_file(FILE *target, int nr, int nq, int np,
                         DATA_TYPE POLYBENCH_3D(A, NR, NQ, NP, nr, nq, np))
{
    int i, j, k;
    for (i = 0; i < nr; i++)
        for (j = 0; j < nq; j++)
            for (k = 0; k < np; k++)
            {
                fprintf(target, DATA_PRINTF_MODIFIER, A[i][j][k]);
                if ((i * nq * np + j * np + k + 1) % 20 == 0)
                    fprintf(target, "\n");
            }
}

/* 打印输出结果 (用于防止DCE) */
static
void print_array(int nr, int nq, int np,
                 DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np))
{
    int i, j, k;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("A");
    for (i = 0; i < nr; i++)
        for (j = 0; j < nq; j++)
            for (k = 0; k < np; k++) {
                if ((i*nq*np + j*np + k) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
                fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
            }
    POLYBENCH_DUMP_END("A");
    POLYBENCH_DUMP_FINISH;
}

/* GPU Kernel 1: 计算 sum[p] = A[r][q][s] * C4[s][p] */
__global__ void doitgen_kernel1(int nr, int nq, int np, DATA_TYPE *A, DATA_TYPE *C4, DATA_TYPE *sum, int r)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y * blockDim.y + threadIdx.y;

    if (p < np && q < nq)
    {
        DATA_TYPE val = SCALAR_VAL(0.0);
        for (int s = 0; s < np; s++)
        {
            val += A[r * (nq * np) + q * np + s] * C4[s * np + p];
        }
        sum[r * (nq * np) + q * np + p] = val;
    }
}

/* GPU Kernel 2: A[r][q][p] = sum[p] */
__global__ void doitgen_kernel2(int nr, int nq, int np, DATA_TYPE *A, DATA_TYPE *sum, int r)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y * blockDim.y + threadIdx.y;

    if (p < np && q < nq)
    {
        A[r * (nq * np) + q * np + p] = sum[r * (nq * np) + q * np + p];
    }
}

/* CUDA 管理函数 */
void doitgenCuda(int nr, int nq, int np,
                 DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
                 DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np))
{
    DATA_TYPE *A_gpu;
    DATA_TYPE *C4_gpu;
    DATA_TYPE *sum_gpu;

    // 分配显存
    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NR * NQ * NP);
    cudaMalloc((void **)&C4_gpu, sizeof(DATA_TYPE) * NP * NP);
    cudaMalloc((void **)&sum_gpu, sizeof(DATA_TYPE) * NR * NQ * NP);

    // 数据传输 HtoD
    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NR * NQ * NP, cudaMemcpyHostToDevice);
    cudaMemcpy(C4_gpu, C4, sizeof(DATA_TYPE) * NP * NP, cudaMemcpyHostToDevice);

    // 设置执行配置
    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((np + block.x - 1) / block.x, (nq + block.y - 1) / block.y);

    /* 启动计时器 */
    polybench_start_instruments;

    // 按层 (r) 进行迭代以匹配串行依赖关系
    for (int r = 0; r < nr; r++)
    {
        doitgen_kernel1<<< grid, block >>>(nr, nq, np, A_gpu, C4_gpu, sum_gpu, r);
        cudaDeviceSynchronize();
        doitgen_kernel2<<< grid, block >>>(nr, nq, np, A_gpu, sum_gpu, r);
        cudaDeviceSynchronize();
    }

    /* 停止计时器 */
    polybench_stop_instruments;
    polybench_print_instruments;

    // 数据传输 DtoH
    cudaMemcpy(A, A_gpu, sizeof(DATA_TYPE) * NR * NQ * NP, cudaMemcpyDeviceToHost);

    // 释放资源
    cudaFree(A_gpu);
    cudaFree(C4_gpu);
    cudaFree(sum_gpu);
}

int main(int argc, char** argv)
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }

    /* 获取问题规模 */
    int nr = NR;
    int nq = NQ;
    int np = NP;

    /* 变量声明 */
    POLYBENCH_3D_ARRAY_DECL(A, DATA_TYPE, NR, NQ, NP, nr, nq, np);
    POLYBENCH_2D_ARRAY_DECL(C4, DATA_TYPE, NP, NP, np, np);

    /* 初始化 */
    init_array(nr, nq, np, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C4));

    /* 设置设备 */
    cudaSetDevice(GPU_DEVICE);

    /* 执行 GPU 计算 */
    doitgenCuda(nr, nq, np, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(C4));

    /* 防止死代码消除并输出 */
    if (verbose)
    {
        FILE *output_file = fopen("c_doitgen_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, nr, nq, np, POLYBENCH_ARRAY(A)));
            fclose(output_file);
        }
    }

    /* 清理内存 */
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(C4);

    return 0;
}
