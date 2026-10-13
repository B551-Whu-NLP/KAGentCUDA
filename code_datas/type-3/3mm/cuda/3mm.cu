/**
 * 3mm.cu: 这是一个 CUDA 版本的 PolyBench 3mm 算法实现。
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>

/* 包含 PolyBench 基础头文件 */
#define POLYBENCH_TIME 1

/* 解决 C/C++ 混合编译下的符号链接问题 */
#include <polybench.h>

#include "3mm.h"

/* 定义 CUDA 线程块大小 */
#ifndef DIM_THREAD_BLOCK_X
#define DIM_THREAD_BLOCK_X 32
#endif
#ifndef DIM_THREAD_BLOCK_Y
#define DIM_THREAD_BLOCK_Y 8
#endif

/* 初始化数组 (保持与原始串行代码逻辑一致) */
static void init_array(int ni, int nj, int nk, int nl, int nm,
                       DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
                       DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
                       DATA_TYPE POLYBENCH_2D(C, NJ, NM, nj, nm),
                       DATA_TYPE POLYBENCH_2D(D, NM, NL, nm, nl))
{
    int i, j;

    for (i = 0; i < ni; i++)
        for (j = 0; j < nk; j++)
            A[i][j] = (DATA_TYPE)((i * j + 1) % ni) / (5 * ni);
    for (i = 0; i < nk; i++)
        for (j = 0; j < nj; j++)
            B[i][j] = (DATA_TYPE)((i * (j + 1) + 2) % nj) / (5 * nj);
    for (i = 0; i < nj; i++)
        for (j = 0; j < nm; j++)
            C[i][j] = (DATA_TYPE)(i * (j + 3) % nl) / (5 * nl);
    for (i = 0; i < nm; i++)
        for (j = 0; j < nl; j++)
            D[i][j] = (DATA_TYPE)((i * (j + 2) + 2) % nk) / (5 * nk);
}

/* 打印结果 (用于防止死代码消除并校验) */
static void print_array(int ni, int nl,
                        DATA_TYPE POLYBENCH_2D(G, NI, NL, ni, nl))
{
    int i, j;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("G");
    for (i = 0; i < ni; i++)
        for (j = 0; j < nl; j++)
        {
            if ((i * ni + j) % 20 == 0)
                fprintf(POLYBENCH_DUMP_TARGET, "\n");
            fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, G[i][j]);
        }
    POLYBENCH_DUMP_END("G");
    POLYBENCH_DUMP_FINISH;
}
static void print_array_to_file(FILE *target, int ni, int nl,
                                DATA_TYPE POLYBENCH_2D(G, NI, NL, ni, nl))
{
    int i, j;
    for (i = 0; i < ni; i++)
        for (j = 0; j < nl; j++)
        {
            fprintf(target, DATA_PRINTF_MODIFIER, G[i][j]);
            if ((i * ni + j + 1) % 20 == 0)
                fprintf(target, "\n");
        }
}

/* GPU Kernel 1: E = A * B */
__global__ void mm3_kernel1(int ni, int nj, int nk, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *E)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ni && j < nj)
    {
        DATA_TYPE val = SCALAR_VAL(0.0);
        for (int k = 0; k < nk; ++k)
        {
            val += A[i * nk + k] * B[k * nj + j];
        }
        E[i * nj + j] = val;
    }
}

/* GPU Kernel 2: F = C * D */
__global__ void mm3_kernel2(int nj, int nl, int nm, DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *F)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nj && j < nl)
    {
        DATA_TYPE val = SCALAR_VAL(0.0);
        for (int k = 0; k < nm; ++k)
        {
            val += C[i * nm + k] * D[k * nl + j];
        }
        F[i * nl + j] = val;
    }
}

/* GPU Kernel 3: G = E * F */
__global__ void mm3_kernel3(int ni, int nl, int nj, DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ni && j < nl)
    {
        DATA_TYPE val = SCALAR_VAL(0.0);
        for (int k = 0; k < nj; ++k)
        {
            val += E[i * nj + k] * F[k * nl + j];
        }
        G[i * nl + j] = val;
    }
}

int main(int argc, char **argv)
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
    int nl = NL;
    int nm = NM;

    /* 变量声明/分配 */
    POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
    POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
    POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
    POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
    POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);

    /* 初始化数组 */
    init_array(ni, nj, nk, nl, nm,
               POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B),
               POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));

    /* GPU 内存分配 */
    DATA_TYPE *A_gpu, *B_gpu, *C_gpu, *D_gpu, *E_gpu, *F_gpu, *G_gpu;
    cudaMalloc((void **)&A_gpu, NI * NK * sizeof(DATA_TYPE));
    cudaMalloc((void **)&B_gpu, NK * NJ * sizeof(DATA_TYPE));
    cudaMalloc((void **)&C_gpu, NJ * NM * sizeof(DATA_TYPE));
    cudaMalloc((void **)&D_gpu, NM * NL * sizeof(DATA_TYPE));
    cudaMalloc((void **)&E_gpu, NI * NJ * sizeof(DATA_TYPE));
    cudaMalloc((void **)&F_gpu, NJ * NL * sizeof(DATA_TYPE));
    cudaMalloc((void **)&G_gpu, NI * NL * sizeof(DATA_TYPE));

    /* 数据传输 Host to Device */
    cudaMemcpy(A_gpu, POLYBENCH_ARRAY(A), NI * NK * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, POLYBENCH_ARRAY(B), NK * NJ * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, POLYBENCH_ARRAY(C), NJ * NM * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(D_gpu, POLYBENCH_ARRAY(D), NM * NL * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    /* 设置网格与线程块 */
    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);

    // Grid 1 为 E (NI x NJ)
    dim3 grid1((nj + block.x - 1) / block.x, (ni + block.y - 1) / block.y);
    // Grid 2 为 F (NJ x NL)
    dim3 grid2((nl + block.x - 1) / block.x, (nj + block.y - 1) / block.y);
    // Grid 3 为 G (NI x NL)
    dim3 grid3((nl + block.x - 1) / block.x, (ni + block.y - 1) / block.y);

    /* 开始计时 */
    polybench_start_instruments;

    /* 执行 Kernel */
    mm3_kernel1<<<grid1, block>>>(ni, nj, nk, A_gpu, B_gpu, E_gpu);
    mm3_kernel2<<<grid2, block>>>(nj, nl, nm, C_gpu, D_gpu, F_gpu);

    // 必须等待 E 和 F 计算完成才能开始计算 G
    cudaDeviceSynchronize();

    mm3_kernel3<<<grid3, block>>>(ni, nl, nj, E_gpu, F_gpu, G_gpu);

    cudaDeviceSynchronize();

    /* 停止并打印计时 */
    polybench_stop_instruments;
    polybench_print_instruments;

    /* 数据传输 Device to Host */
    cudaMemcpy(POLYBENCH_ARRAY(G), G_gpu, NI * NL * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    /* 防止死代码消除 (DCE) */
    if (verbose)
    {
        FILE *output_file = fopen("cuda_3mm_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, ni, nl, POLYBENCH_ARRAY(G)));
            fclose(output_file);
        }
    }

    /* 清理显存 */
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    cudaFree(D_gpu);
    cudaFree(E_gpu);
    cudaFree(F_gpu);
    cudaFree(G_gpu);

    /* 清理内存 */
    POLYBENCH_FREE_ARRAY(E);
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(F);
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(D);
    POLYBENCH_FREE_ARRAY(G);

    return 0;
}