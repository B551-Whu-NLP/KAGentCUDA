/**
 * cholesky.cu: CUDA implementation of PolyBench Cholesky Decomposition
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <string.h>

#define POLYBENCH_TIME 1
#include <polybench.h>
#include "cholesky.h"

#ifndef DIM_THREAD_BLOCK_X
# define DIM_THREAD_BLOCK_X 32
#endif

#define GPU_DEVICE 0

/* 数组初始化：严格对应 PolyBench 2016 逻辑，生成正定矩阵 */
static void init_array(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
    int i, j, r, s, t;
    for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++)
            A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
        for (j = i + 1; j < n; j++)
            A[i][j] = 0;
        A[i][i] = 1;
    }

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
        for (j = 0; j <= i; j++)
        {
            fprintf(target, DATA_PRINTF_MODIFIER, A[i][j]);
            if ((i * n + j + 1) % 20 == 0)
                fprintf(target, "\n");
        }
}

/* 结果打印：仅打印下三角 */
static void print_array(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
    int i, j;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("A");
    for (i = 0; i < n; i++)
        for (j = 0; j <= i; j++) {
            if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
            fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
        }
    POLYBENCH_DUMP_END("A");
    POLYBENCH_DUMP_FINISH;
}

/* 内核 1：并行处理第 i 行中 j < i 的所有元素 */
__global__ void cholesky_kernel_column_update(int n, DATA_TYPE *a, int j)
{
    // 并行处理所有行 i > j
    int i = blockIdx.x * blockDim.x + threadIdx.x + (j + 1);

    if (i < n) {
        DATA_TYPE acc = a[i * n + j];
        for (int k = 0; k < j; k++) {
            acc -= a[i * n + k] * a[j * n + k];
        }
        a[i * n + j] = acc / a[j * n + j];
    }
}

__global__ void cholesky_kernel_diag_update(int n, DATA_TYPE *a, int j)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        DATA_TYPE acc = a[j * n + j];
        for (int k = 0; k < j; k++) {
            acc -= a[j * n + k] * a[j * n + k];
        }
        a[j * n + j] = SQRT_FUN(acc > 0 ? acc : SCALAR_VAL(0.0));
    }
}

void choleskyCuda(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
    DATA_TYPE *A_gpu;
    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * N * N);
    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);

    polybench_start_instruments;

    // 外部循环控制行 i 的进度 (不可避免的串行)
    for (int j = 0; j < n; j++) {
        // 1. 先算对角线 (串行)
        cholesky_kernel_diag_update<<<1, 1>>>(n, A_gpu, j);
        cudaDeviceSynchronize();

        // 2. 并行计算该列下方所有的行元素 (并行 i)
        int num_rows = n - (j + 1);
        if (num_rows > 0) {
            int threads = 256;
            int blocks = (num_rows + threads - 1) / threads;
            cholesky_kernel_column_update<<<blocks, threads>>>(n, A_gpu, j);
            cudaDeviceSynchronize();
        }
    }

    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(A, A_gpu, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost);
    cudaFree(A_gpu);
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
    choleskyCuda(n, POLYBENCH_ARRAY(A));

    if (verbose)
    {
        FILE *output_file = fopen("c_cholesky_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(A)));
            fclose(output_file);
        }
    }

    POLYBENCH_FREE_ARRAY(A);
    return 0;
}
