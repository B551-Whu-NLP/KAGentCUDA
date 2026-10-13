/**
 * jacobi2D.cu: CUDA implementation of PolyBench Jacobi-2D stencil
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
#include "jacobi-2d.h"

/* 定义线程块大小 */
#ifndef DIM_THREAD_BLOCK_X
# define DIM_THREAD_BLOCK_X 32
#endif
#ifndef DIM_THREAD_BLOCK_Y
# define DIM_THREAD_BLOCK_Y 8
#endif

#define GPU_DEVICE 0

/* 数组初始化：严格对应 2016 C 串行逻辑 */
static void init_array (int n,
                DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
                DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
            B[i][j] = ((DATA_TYPE) i*(j+3) + 3) / n;
        }
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
static void print_array(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
    int i, j;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("A");
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
            fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
        }
    POLYBENCH_DUMP_END("A");
    POLYBENCH_DUMP_FINISH;
}

/* GPU Kernel 1: B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j]) */
__global__ void jacobi_kernel1(int n, DATA_TYPE *A, DATA_TYPE *B)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < (n - 1) && j >= 1 && j < (n - 1)) {
        B[i * n + j] = SCALAR_VAL(0.2) * (A[i * n + j] + A[i * n + (j - 1)] + 
                       A[i * n + (j + 1)] + A[(i + 1) * n + j] + A[(i - 1) * n + j]);
    }
}

/* GPU Kernel 2: A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][j+1] + B[i+1][j] + B[i-1][j]) */
__global__ void jacobi_kernel2(int n, DATA_TYPE *A, DATA_TYPE *B)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < (n - 1) && j >= 1 && j < (n - 1)) {
        A[i * n + j] = SCALAR_VAL(0.2) * (B[i * n + j] + B[i * n + (j - 1)] + 
                       B[i * n + (j + 1)] + B[(i + 1) * n + j] + B[(i - 1) * n + j]);
    }
}

void jacobi2DCuda(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
    DATA_TYPE *A_gpu, *B_gpu;
    cudaMalloc(&A_gpu, n * n * sizeof(DATA_TYPE));
    cudaMalloc(&B_gpu, n * n * sizeof(DATA_TYPE));

    cudaMemcpy(A_gpu, A, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, n * n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    polybench_start_instruments;

    for (int t = 0; t < tsteps; t++) {
        // 第一阶段: A -> B
        jacobi_kernel1<<<grid, block>>>(n, A_gpu, B_gpu);
        // 第二阶段: B -> A
        jacobi_kernel2<<<grid, block>>>(n, A_gpu, B_gpu);
    }
    
    cudaDeviceSynchronize();

    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(A, A_gpu, n * n * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    cudaFree(A_gpu);
    cudaFree(B_gpu);
}

int main(int argc, char** argv)
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }

    int n = N;
    int tsteps = TSTEPS;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

    init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

    cudaSetDevice(GPU_DEVICE);
    jacobi2DCuda(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

    if (verbose)
    {
        FILE *output_file = fopen("c_jacobi2d_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(A)));
            fclose(output_file);
        }
    }

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);

    return 0;
}
