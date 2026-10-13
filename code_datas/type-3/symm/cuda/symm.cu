/**
 * symm.cu: CUDA implementation of PolyBench SYMM
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <string.h>

#define POLYBENCH_TIME 1
#include <polybench.h>
#include "symm.h"

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
                DATA_TYPE *beta,
                DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
                DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
                DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
    int i, j;
    *alpha = 1.5;
    *beta = 1.2;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++) {
            C[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
            B[i][j] = (DATA_TYPE) ((n+i-j) % 100) / m;
        }
    for (i = 0; i < m; i++) {
        for (j = 0; j <= i; j++)
            A[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
        for (j = i+1; j < m; j++)
            A[i][j] = -999.0; 
    }
}

static void print_array_to_file(FILE *target, int m, int n,
                                DATA_TYPE POLYBENCH_2D(C, M, N, m, n))
{
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++) {
            fprintf(target, DATA_PRINTF_MODIFIER, C[i][j]);
            if ((i * m + j + 1) % 20 == 0) fprintf(target, "\n");
        }
}

/* 结果打印 */
static void print_array(int m, int n, DATA_TYPE POLYBENCH_2D(C,M,N,m,n))
{
    int i, j;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("C");
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++) {
            if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
            fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
        }
    POLYBENCH_DUMP_END("C");
    POLYBENCH_DUMP_FINISH;
}

/* SYMM CUDA Kernel */
__global__ void symm_kernel(int m, int n, DATA_TYPE alpha, DATA_TYPE beta, 
                            DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n)
    {
        DATA_TYPE acc = 0;
        
        for (int k = 0; k < m; k++) {
            if (k < i) {
                // A 是下三角存储，k < i 时访问 A[i][k]
                acc += alpha * b[k * n + j] * a[i * m + k];
            } else if (k == i) {
                // 对角线
                acc += alpha * b[i * n + j] * a[i * m + i];
            } else {
                // 利用对称性访问 A[k][i]
                acc += alpha * b[k * n + j] * a[k * m + i];
            }
        }
        c[i * n + j] = beta * c[i * n + j] + acc;
    }
}

void symmCuda(int m, int n, DATA_TYPE alpha, DATA_TYPE beta, 
              DATA_TYPE POLYBENCH_2D(C,M,N,m,n), 
              DATA_TYPE POLYBENCH_2D(A,M,M,m,m), 
              DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
    DATA_TYPE *A_gpu, *B_gpu, *C_gpu;

    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * M * M);
    cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * M * N);
    cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * M * N);

    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * M * M, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * M * N, cudaMemcpyHostToDevice);

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    polybench_start_instruments;

    symm_kernel<<<grid, block>>>(m, n, alpha, beta, A_gpu, B_gpu, C_gpu);
    
    cudaDeviceSynchronize();

    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(C, C_gpu, sizeof(DATA_TYPE) * M * N, cudaMemcpyDeviceToHost);

    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
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
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, M, N, m, n);
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, M, m, m);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, M, N, m, n);

    init_array(m, n, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

    cudaSetDevice(GPU_DEVICE);
    symmCuda(m, n, alpha, beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

    if (verbose)
    {
        FILE *output_file = fopen("c_symm_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, m, n, POLYBENCH_ARRAY(C)));
            fclose(output_file);
        }
    }

    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);

    return 0;
}
