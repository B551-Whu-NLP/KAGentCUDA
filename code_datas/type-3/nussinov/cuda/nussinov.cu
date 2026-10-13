/**
 * nussinov.cu: CUDA implementation of Nussinov algorithm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <string.h>

#define POLYBENCH_TIME 1
#include <polybench.h>
#include "nussinov.h"

typedef char base;

#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0)

#ifndef DIM_THREAD_BLOCK_X
# define DIM_THREAD_BLOCK_X 256
#endif

#define GPU_DEVICE 0

/* 数组初始化 */
static void init_array(int n, base POLYBENCH_1D(seq,N,n), DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
    int i, j;
    for (i = 0; i < n; i++) {
        seq[i] = (base)((i + 1) % 4);
    }
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            table[i][j] = 0;
}

static void print_array_to_file(FILE *target, int n,
                                DATA_TYPE POLYBENCH_2D(table, N, N, n, n))
{
    int i, j;
    int t = 0;
    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            fprintf(target, DATA_PRINTF_MODIFIER, table[i][j]);
            if ((t + 1) % 20 == 0) fprintf(target, "\n");
            t++;
        }
    }
}

/* 结果打印 */
static void print_array(int n, DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
    int i, j;
    int t = 0;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("table");
    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            if (t % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
            fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, table[i][j]);
            t++;
        }
    }
    POLYBENCH_DUMP_END("table");
    POLYBENCH_DUMP_FINISH;
}

/* CUDA Kernel: 按对角线并行计算 */
__global__ void nussinov_kernel(int n, base *seq, DATA_TYPE *table, int d)
{
    // i 表示起始行，i 从 0 到 n-d-1
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n - d)
    {
        int j = i + d;
        DATA_TYPE res = table[i * n + j];

        // 1. table[i][j] = max(table[i][j], table[i][j-1])
        DATA_TYPE val1 = table[i * n + j - 1];
        if (val1 > res) res = val1;

        // 2. table[i][j] = max(table[i][j], table[i+1][j])
        DATA_TYPE val2 = table[(i + 1) * n + j];
        if (val2 > res) res = val2;

        // 3. table[i+1][j-1] + match
        if (i + 1 < n && j - 1 >= 0) {
            DATA_TYPE m = (i < j - 1) ? (table[(i + 1) * n + j - 1] + match(seq[i], seq[j])) : table[(i + 1) * n + j - 1];
            if (m > res) res = m;
        }

        // 4. 分割点 k 计算 (k = i+1 ... j-1)
        for (int k = i + 1; k < j; k++) {
            DATA_TYPE val_k = table[i * n + k] + table[(k + 1) * n + j];
            if (val_k > res) res = val_k;
        }

        table[i * n + j] = res;
    }
}

void nussinovCuda(int n, base POLYBENCH_1D(seq,N,n), DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
    base *seq_gpu;
    DATA_TYPE *table_gpu;

    cudaMalloc(&seq_gpu, n * sizeof(base));
    cudaMalloc(&table_gpu, n * n * sizeof(DATA_TYPE));

    cudaMemcpy(seq_gpu, seq, n * sizeof(base), cudaMemcpyHostToDevice);
    cudaMemcpy(table_gpu, table, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 block(DIM_THREAD_BLOCK_X);

    polybench_start_instruments;

    // d 为对角线偏移量，控制计算顺序
    for (int d = 1; d < n; d++) {
        // 对角线上有多少个元素需要计算？ n-d 个
        dim3 grid((n - d + block.x - 1) / block.x);
        nussinov_kernel<<<grid, block>>>(n, seq_gpu, table_gpu, d);
        // 必须同步，因为下一条更长的对角线依赖当前结果
        cudaDeviceSynchronize();
    }

    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(table, table_gpu, n * n * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    cudaFree(seq_gpu);
    cudaFree(table_gpu);
}

int main(int argc, char** argv)
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }

    int n = N;
    POLYBENCH_1D_ARRAY_DECL(seq, base, N, n);
    POLYBENCH_2D_ARRAY_DECL(table, DATA_TYPE, N, N, n, n);

    init_array(n, POLYBENCH_ARRAY(seq), POLYBENCH_ARRAY(table));

    cudaSetDevice(GPU_DEVICE);
    nussinovCuda(n, POLYBENCH_ARRAY(seq), POLYBENCH_ARRAY(table));

    if (verbose)
    {
        FILE *output_file = fopen("c_nussinov_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(table)));
            fclose(output_file);
        }
    }

    POLYBENCH_FREE_ARRAY(seq);
    POLYBENCH_FREE_ARRAY(table);

    return 0;
}
