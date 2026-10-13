/**
 * lu.cu: CUDA implementation of PolyBench LU decomposition
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
#include "lu.h"

/* 定义线程块大小 */
#ifndef DIM_THREAD_BLOCK_X
# define DIM_THREAD_BLOCK_X 32
#endif
#ifndef DIM_THREAD_BLOCK_Y
# define DIM_THREAD_BLOCK_Y 8
#endif

#define GPU_DEVICE 0

/* 数组初始化：严格对应 2016 C 版正定矩阵初始化逻辑 */
static
void init_array (int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
    int i, j, r, s, t;
    // 临时生成一个下三角矩阵
    for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++)
            A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
        for (j = i + 1; j < n; j++)
            A[i][j] = 0;
        A[i][i] = 1;
    }

    // A = A * A^T 使其成为正定矩阵，确保 LU 分解不遇零主元
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
    for (r = 0; r < n; ++r)
        for (s = 0; s < n; ++s)
            (POLYBENCH_ARRAY(B))[r][s] = 0;
            
    for (t = 0; t < n; ++t)
        for (r = 0; r < n; ++r)
            for (s = 0; s < n; ++s)
                (POLYBENCH_ARRAY(B))[r][s] += A[r][t] * A[s][t];
                
    for (r = 0; r < n; ++r)
        for (s = 0; s < n; ++s)
            A[r][s] = (POLYBENCH_ARRAY(B))[r][s];
            
    POLYBENCH_FREE_ARRAY(B);
}

static void print_array_to_file(FILE *target, int n,
                                DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            fprintf(target, DATA_PRINTF_MODIFIER, A[i][j]);
            if ((i * n + j + 1) % 20 == 0) fprintf(target, "\n");
        }
}

/* 结果打印：对应 PolyBench 2016 DUMP 规范 */
static
void print_array(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
    int i, j;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("A");
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
            fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
        }
    POLYBENCH_DUMP_END("A");
    POLYBENCH_DUMP_FINISH;
}

/* Kernel 1: 计算当前步的 L 矩阵分量 (归一化第 k 行) */
__global__ void lu_kernel1(int n, DATA_TYPE *A, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int target_j = k + 1 + j;

    if (target_j < n)
    {
        A[k * n + target_j] /= A[k * n + k];
    }
}

/* Kernel 2: 更新剩余的 Schur 补阵 (A = A - L * U) */
__global__ void lu_kernel2(int n, DATA_TYPE *A, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    int target_i = k + 1 + i;
    int target_j = k + 1 + j;

    if (target_i < n && target_j < n)
    {
        // A[i][j] -= A[i][k] * A[k][j]
        A[target_i * n + target_j] -= A[target_i * n + k] * A[k * n + target_j];
    }
}

void luCuda(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
    DATA_TYPE* AGpu;
    cudaMalloc(&AGpu, N * N * sizeof(DATA_TYPE));
    cudaMemcpy(AGpu, A, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 block1(DIM_THREAD_BLOCK_X);
    dim3 block2(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);

    /* 启动计时 */
    polybench_start_instruments;

    for (int k = 0; k < n; k++)
    {
        // 1. 处理第 k 行
        int size_k1 = n - (k + 1);
        if (size_k1 > 0) {
            dim3 grid1((size_k1 + block1.x - 1) / block1.x);
            lu_kernel1<<<grid1, block1>>>(n, AGpu, k);
        }
        
        // 2. 更新下方矩阵
        int size_k2 = n - (k + 1);
        if (size_k2 > 0) {
            dim3 grid2((size_k2 + block2.x - 1) / block2.x, 
                       (size_k2 + block2.y - 1) / block2.y);
            lu_kernel2<<<grid2, block2>>>(n, AGpu, k);
        }
    }
    
    cudaDeviceSynchronize();

    /* 停止计时 */
    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(A, AGpu, N * N * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaFree(AGpu);
}

int main(int argc, char** argv)
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }

    int n = N;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

    init_array(n, POLYBENCH_ARRAY(A));

    cudaSetDevice(GPU_DEVICE);
    luCuda(n, POLYBENCH_ARRAY(A));

    if (verbose)
    {
        FILE *output_file = fopen("c_lu_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(A)));
            fclose(output_file);
        }
    }

    POLYBENCH_FREE_ARRAY(A);

    return 0;
}
