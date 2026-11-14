/**
 * adi.cu: CUDA implementation of PolyBench ADI (Alternating Direction Implicit solver)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>

#define POLYBENCH_TIME 1
#include <polybench.h>
#include "adi.h"

#ifndef DIM_THREAD_BLOCK_X
#define DIM_THREAD_BLOCK_X 256
#endif

#define GPU_DEVICE 0

/* 数组初始化：对应 PolyBench 2016 逻辑 */
static void init_array(int n, DATA_TYPE POLYBENCH_2D(u, N, N, n, n))
{
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            u[i][j] = (DATA_TYPE)(i + n - j) / n;
}
static void print_array_to_file(FILE *target, int n,
                                DATA_TYPE POLYBENCH_2D(u, N, N, n, n))
{
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            fprintf(target, DATA_PRINTF_MODIFIER, u[i][j]);
            if ((i * n + j + 1) % 20 == 0)
                fprintf(target, "\n");
        }
}
/* 结果打印：对应 PolyBench 2016 DUMP 规范 */
static void print_array(int n, DATA_TYPE POLYBENCH_2D(u, N, N, n, n))
{
    int i, j;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("u");
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            if ((i * n + j) % 20 == 0)
                fprintf(POLYBENCH_DUMP_TARGET, "\n");
            fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, u[i][j]);
        }
    POLYBENCH_DUMP_END("u");
    POLYBENCH_DUMP_FINISH;
}

/* Kernel 1: Column Sweep (针对每一列进行前向消元和后向代换) */
__global__ void adi_kernel_column_sweep(int n, int t, DATA_TYPE a, DATA_TYPE b, DATA_TYPE c,
                                        DATA_TYPE d, DATA_TYPE e, DATA_TYPE f,
                                        DATA_TYPE *u, DATA_TYPE *v, DATA_TYPE *p, DATA_TYPE *q)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i < _PB_N - 1)
    {
        v[0 * n + i] = SCALAR_VAL(1.0);
        p[i * n + 0] = SCALAR_VAL(0.0);
        q[i * n + 0] = v[0 * n + i];

        for (int j = 1; j < _PB_N - 1; j++)
        {
            p[i * n + j] = -c / (a * p[i * n + j - 1] + b);
            q[i * n + j] = (-d * u[j * n + i - 1] + (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * d) * u[j * n + i] - f * u[j * n + i + 1] - a * q[i * n + j - 1]) / (a * p[i * n + j - 1] + b);
        }

        v[(_PB_N - 1) * n + i] = SCALAR_VAL(1.0);
        for (int j = _PB_N - 2; j >= 1; j--)
        {
            v[j * n + i] = p[i * n + j] * v[(j + 1) * n + i] + q[i * n + j];
        }
    }
}

/* Kernel 2: Row Sweep (针对每一行进行前向消元和后向代换) */
__global__ void adi_kernel_row_sweep(int n, int t, DATA_TYPE a, DATA_TYPE b, DATA_TYPE c,
                                     DATA_TYPE d, DATA_TYPE e, DATA_TYPE f,
                                     DATA_TYPE *u, DATA_TYPE *v, DATA_TYPE *p, DATA_TYPE *q)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i < _PB_N - 1)
    {
        u[i * n + 0] = SCALAR_VAL(1.0);
        p[i * n + 0] = SCALAR_VAL(0.0);
        q[i * n + 0] = u[i * n + 0];

        for (int j = 1; j < _PB_N - 1; j++)
        {
            p[i * n + j] = -f / (d * p[i * n + j - 1] + e);
            q[i * n + j] = (-a * v[(i - 1) * n + j] + (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * a) * v[i * n + j] - c * v[(i + 1) * n + j] - d * q[i * n + j - 1]) / (d * p[i * n + j - 1] + e);
        }

        u[i * n + (_PB_N - 1)] = SCALAR_VAL(1.0);
        for (int j = _PB_N - 2; j >= 1; j--)
        {
            u[i * n + j] = p[i * n + j] * u[i * n + j + 1] + q[i * n + j];
        }
    }
}

void adiCuda(int tsteps, int n, DATA_TYPE POLYBENCH_2D(u, N, N, n, n), DATA_TYPE POLYBENCH_2D(v, N, N, n, n),
             DATA_TYPE POLYBENCH_2D(p, N, N, n, n), DATA_TYPE POLYBENCH_2D(q, N, N, n, n))
{
    DATA_TYPE *u_gpu, *v_gpu, *p_gpu, *q_gpu;
    cudaMalloc(&u_gpu, N * N * sizeof(DATA_TYPE));
    cudaMalloc(&v_gpu, N * N * sizeof(DATA_TYPE));
    cudaMalloc(&p_gpu, N * N * sizeof(DATA_TYPE));
    cudaMalloc(&q_gpu, N * N * sizeof(DATA_TYPE));

    cudaMemcpy(u_gpu, u, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    // 计算数学常量
    DATA_TYPE DX = SCALAR_VAL(1.0) / (DATA_TYPE)_PB_N;
    DATA_TYPE DY = SCALAR_VAL(1.0) / (DATA_TYPE)_PB_N;
    DATA_TYPE DT = SCALAR_VAL(1.0) / (DATA_TYPE)_PB_TSTEPS;
    DATA_TYPE B1 = SCALAR_VAL(2.0);
    DATA_TYPE B2 = SCALAR_VAL(1.0);
    DATA_TYPE mul1 = B1 * DT / (DX * DX);
    DATA_TYPE mul2 = B2 * DT / (DY * DY);
    DATA_TYPE a = -mul1 / SCALAR_VAL(2.0);
    DATA_TYPE b = SCALAR_VAL(1.0) + mul1;
    DATA_TYPE c = a;
    DATA_TYPE d = -mul2 / SCALAR_VAL(2.0);
    DATA_TYPE e = SCALAR_VAL(1.0) + mul2;
    DATA_TYPE f = d;

    dim3 block(DIM_THREAD_BLOCK_X);
    dim3 grid((n + block.x - 1) / block.x);

    polybench_start_instruments;

    for (int t = 1; t <= _PB_TSTEPS; t++)
    {
        // 1. Column Sweep
        adi_kernel_column_sweep<<<grid, block>>>(n, t, a, b, c, d, e, f, u_gpu, v_gpu, p_gpu, q_gpu);
        cudaDeviceSynchronize();

        // 2. Row Sweep
        adi_kernel_row_sweep<<<grid, block>>>(n, t, a, b, c, d, e, f, u_gpu, v_gpu, p_gpu, q_gpu);
        cudaDeviceSynchronize();
    }

    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(u, u_gpu, N * N * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    cudaFree(u_gpu);
    cudaFree(v_gpu);
    cudaFree(p_gpu);
    cudaFree(q_gpu);
}

int main(int argc, char **argv)
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }
    int n = N;
    int tsteps = TSTEPS;

    POLYBENCH_2D_ARRAY_DECL(u, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(v, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(p, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(q, DATA_TYPE, N, N, n, n);

    init_array(n, POLYBENCH_ARRAY(u));

    cudaSetDevice(GPU_DEVICE);
    adiCuda(tsteps, n, POLYBENCH_ARRAY(u), POLYBENCH_ARRAY(v), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));

    if (verbose)
    {
        FILE *output_file = fopen("cuda_adi_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(u)));
            fclose(output_file);
        }
    }

    POLYBENCH_FREE_ARRAY(u);
    POLYBENCH_FREE_ARRAY(v);
    POLYBENCH_FREE_ARRAY(p);
    POLYBENCH_FREE_ARRAY(q);

    return 0;
}
