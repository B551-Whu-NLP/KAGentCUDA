#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <string.h>

#define POLYBENCH_TIME 1
#include <polybench.h>
#include "trmm.h"

#ifndef DIM_THREAD_BLOCK_X
# define DIM_THREAD_BLOCK_X 32
#endif
#ifndef DIM_THREAD_BLOCK_Y
# define DIM_THREAD_BLOCK_Y 8
#endif

#define GPU_DEVICE 0

/* 数组初始化：严格对应 PolyBench 2016 逻辑 */
static void init_array(int m, int n,
                DATA_TYPE *alpha,
                DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
                DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
    int i, j;
    *alpha = 1.5;
    for (i = 0; i < m; i++) {
        for (j = 0; j < i; j++) {
            A[i][j] = (DATA_TYPE)((i+j) % m) / m;
        }
        A[i][i] = 1.0;
        for (j = 0; j < n; j++) {
            B[i][j] = (DATA_TYPE)((n+(i-j)) % n) / n;
        }
    }
}

static void print_array_to_file(FILE *target, int m, int n,
                                DATA_TYPE POLYBENCH_2D(B, M, N, m, n))
{
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++) {
            fprintf(target, DATA_PRINTF_MODIFIER, B[i][j]);
            if ((i * m + j + 1) % 20 == 0) fprintf(target, "\n");
        }
}

/* 结果打印 */
static void print_array(int m, int n, DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
    int i, j;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("B");
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++) {
            if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
            fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, B[i][j]);
        }
    POLYBENCH_DUMP_END("B");
    POLYBENCH_DUMP_FINISH;
}

/* TRMM CUDA Kernel */
__global__ void trmm_kernel(int m, int n, DATA_TYPE alpha, DATA_TYPE *a, DATA_TYPE *b_in, DATA_TYPE *b_out)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n)
    {
        // 必须从未修改的 b_in 中读取数据
        DATA_TYPE acc = b_in[i * n + j];
        
        for (int k = i + 1; k < m; k++) {
            acc += a[k * m + i] * b_in[k * n + j];
        }
        b_out[i * n + j] = alpha * acc;
    }
}

void trmmCuda(int m, int n, DATA_TYPE alpha, 
              DATA_TYPE POLYBENCH_2D(A,M,M,m,m), 
              DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
    DATA_TYPE *A_gpu, *B_in_gpu, *B_out_gpu;

    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * M * M);
    cudaMalloc((void **)&B_in_gpu, sizeof(DATA_TYPE) * M * N);
    cudaMalloc((void **)&B_out_gpu, sizeof(DATA_TYPE) * M * N);

    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * M * M, cudaMemcpyHostToDevice);
    cudaMemcpy(B_in_gpu, B, sizeof(DATA_TYPE) * M * N, cudaMemcpyHostToDevice);
    // B_out_gpu 初始也可以拷贝一份，或者直接由 Kernel 覆盖
    cudaMemcpy(B_out_gpu, B, sizeof(DATA_TYPE) * M * N, cudaMemcpyHostToDevice);

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    polybench_start_instruments;

    // 传入两个 B 指针
    trmm_kernel<<<grid, block>>>(m, n, alpha, A_gpu, B_in_gpu, B_out_gpu);
    
    cudaDeviceSynchronize();

    polybench_stop_instruments;
    polybench_print_instruments;

    // 从 B_out_gpu 拷贝回结果
    cudaMemcpy(B, B_out_gpu, sizeof(DATA_TYPE) * M * N, cudaMemcpyDeviceToHost);

    cudaFree(A_gpu);
    cudaFree(B_in_gpu);
    cudaFree(B_out_gpu);
}

int main(int argc, char** argv)
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }

    int m = M;
    int n = N;

    DATA_TYPE alpha;
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, M, m, m);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, M, N, m, n);

    init_array(m, n, &alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

    cudaSetDevice(GPU_DEVICE);
    trmmCuda(m, n, alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

    if (verbose)
    {
        FILE *output_file = fopen("cuda_trmm_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, m, n, POLYBENCH_ARRAY(B)));
            fclose(output_file);
        }
    }

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);

    return 0;
}