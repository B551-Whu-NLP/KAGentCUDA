/**
 * heat-3d.cu: CUDA implementation of 3D Heat Equation
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <string.h>

#define POLYBENCH_TIME 1
#include <polybench.h>
#include "heat-3d.h"

/* 定义 3D 线程块大小 (建议根据 GPU 架构调整，确保 block.x * block.y * block.z <= 1024) */
#ifndef DIM_THREAD_BLOCK_X
# define DIM_THREAD_BLOCK_X 32
#endif
#ifndef DIM_THREAD_BLOCK_Y
# define DIM_THREAD_BLOCK_Y 8
#endif
#ifndef DIM_THREAD_BLOCK_Z
# define DIM_THREAD_BLOCK_Z 4
#endif

#define GPU_DEVICE 0

/* 数组初始化：严格对应 PolyBench 2016 逻辑 */
static void init_array (int n,
                DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
                DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
    int i, j, k;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            for (k = 0; k < n; k++)
                A[i][j][k] = B[i][j][k] = (DATA_TYPE) (i + j + (n-k))* 10 / (n);
}

static void print_array_to_file(FILE *target, int n,
                                DATA_TYPE POLYBENCH_3D(A, N, N, N, n, n, n))
{
    int i, j, k;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            for (k = 0; k < n; k++) {
                fprintf(target, DATA_PRINTF_MODIFIER, A[i][j][k]);
                if ((i * n * n + j * n + k + 1) % 20 == 0) fprintf(target, "\n");
            }
}

/* 结果打印 */
static void print_array(int n, DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n))
{
    int i, j, k;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("A");
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            for (k = 0; k < n; k++) {
                if ((i * n * n + j * n + k) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
                fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
            }
    POLYBENCH_DUMP_END("A");
    POLYBENCH_DUMP_FINISH;
}

/* CUDA Kernel: 7点模板计算 */
__global__ void heat3d_kernel(int n, DATA_TYPE *A, DATA_TYPE *B)
{
    // 映射 3D 线程索引
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    // 仅更新边界内部的点 (1 到 n-2)
    if (i >= 1 && i < n - 1 && j >= 1 && j < n - 1 && k >= 1 && k < n - 1)
    {
        B[i*n*n + j*n + k] = SCALAR_VAL(0.125) * (A[(i+1)*n*n + j*n + k] - SCALAR_VAL(2.0) * A[i*n*n + j*n + k] + A[(i-1)*n*n + j*n + k])
                           + SCALAR_VAL(0.125) * (A[i*n*n + (j+1)*n + k] - SCALAR_VAL(2.0) * A[i*n*n + j*n + k] + A[i*n*n + (j-1)*n + k])
                           + SCALAR_VAL(0.125) * (A[i*n*n + j*n + k + 1] - SCALAR_VAL(2.0) * A[i*n*n + j*n + k] + A[i*n*n + j*n + k - 1])
                           + A[i*n*n + j*n + k];
    }
}

void heat3dCuda(int tsteps, int n, DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n), DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
    DATA_TYPE *A_gpu, *B_gpu;
    size_t size = (size_t)n * n * n * sizeof(DATA_TYPE);

    cudaMalloc(&A_gpu, size);
    cudaMalloc(&B_gpu, size);
    cudaMemcpy(A_gpu, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, size, cudaMemcpyHostToDevice);

    // 设置 3D 网格和线程块
    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y, DIM_THREAD_BLOCK_Z);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y, (n + block.z - 1) / block.z);

    polybench_start_instruments;

    for (int t = 1; t <= tsteps; t++)
    {
        // 第一步: A -> B
        heat3d_kernel<<<grid, block>>>(n, A_gpu, B_gpu);
        // 第二步: B -> A
        heat3d_kernel<<<grid, block>>>(n, B_gpu, A_gpu);
    }
    
    cudaDeviceSynchronize();

    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(A, A_gpu, size, cudaMemcpyDeviceToHost);

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

    POLYBENCH_3D_ARRAY_DECL(A, DATA_TYPE, N, N, N, n, n, n);
    POLYBENCH_3D_ARRAY_DECL(B, DATA_TYPE, N, N, N, n, n, n);

    init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

    cudaSetDevice(GPU_DEVICE);
    heat3dCuda(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

    if (verbose)
    {
        FILE *output_file = fopen("c_heat3d_output.txt", "w");
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
