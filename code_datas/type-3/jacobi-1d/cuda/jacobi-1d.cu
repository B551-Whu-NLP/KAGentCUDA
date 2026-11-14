/**
 * jacobi1D.cu: CUDA implementation of PolyBench Jacobi-1D stencil
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <string.h>

#define POLYBENCH_TIME 1
#include <polybench.h>
#include "jacobi-1d.h"

#ifndef DIM_THREAD_BLOCK_X
# define DIM_THREAD_BLOCK_X 256
#endif

#define GPU_DEVICE 0

/* 数组初始化：严格对应 2016 C 串行逻辑 */
static void init_array(int n, DATA_TYPE POLYBENCH_1D(A,N,n), DATA_TYPE POLYBENCH_1D(B,N,n))
{
    int i;
    for (i = 0; i < n; i++) {
        A[i] = ((DATA_TYPE) i + 2) / n;
        B[i] = ((DATA_TYPE) i + 3) / n;
    }
}

static void print_array_to_file(FILE *target, int n,
                                DATA_TYPE POLYBENCH_1D(A, N, n))
{
    int i;
    for (i = 0; i < n; i++) {
        fprintf(target, DATA_PRINTF_MODIFIER, A[i]);
        if ((i + 1) % 20 == 0) fprintf(target, "\n");
    }
}

/* 结果打印：对应 PolyBench 2016 DUMP 规范 */
static void print_array(int n, DATA_TYPE POLYBENCH_1D(A,N,n))
{
    int i;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("A");
    for (i = 0; i < n; i++) {
        if (i % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
        fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i]);
    }
    POLYBENCH_DUMP_END("A");
    POLYBENCH_DUMP_FINISH;
}

/* GPU Kernel 1: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1]) */
__global__ void jacobi_kernel1(int n, DATA_TYPE *A, DATA_TYPE *B)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < (n - 1)) {
        B[i] = 0.33333f * (A[i-1] + A[i] + A[i+1]);
    }
}

/* GPU Kernel 2: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1]) */
__global__ void jacobi_kernel2(int n, DATA_TYPE *A, DATA_TYPE *B)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < (n - 1)) {
        A[i] = 0.33333f * (B[i-1] + B[i] + B[i+1]);
    }
}

void jacobi1DCuda(int tsteps, int n, DATA_TYPE POLYBENCH_1D(A,N,n), DATA_TYPE POLYBENCH_1D(B,N,n))
{
    DATA_TYPE *A_gpu, *B_gpu;
    cudaMalloc(&A_gpu, n * sizeof(DATA_TYPE));
    cudaMalloc(&B_gpu, n * sizeof(DATA_TYPE));

    cudaMemcpy(A_gpu, A, n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 block(DIM_THREAD_BLOCK_X);
    dim3 grid((n + block.x - 1) / block.x);

    polybench_start_instruments;

    for (int t = 0; t < tsteps; t++) {
        // Step 1: A -> B
        jacobi_kernel1<<<grid, block>>>(n, A_gpu, B_gpu);
        cudaDeviceSynchronize();
        // Step 2: B -> A
        jacobi_kernel2<<<grid, block>>>(n, A_gpu, B_gpu);
        cudaDeviceSynchronize();
    }

    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(A, A_gpu, n * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

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

    POLYBENCH_1D_ARRAY_DECL(A, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(B, DATA_TYPE, N, n);

    init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

    cudaSetDevice(GPU_DEVICE);
    jacobi1DCuda(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

    if (verbose)
    {
        FILE *output_file = fopen("c_jacobi1d_output.txt", "w");
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
