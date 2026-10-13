/**
 * ludcmp.cu: CUDA implementation of PolyBench LUDCMP
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <string.h>

#define POLYBENCH_TIME 1
#include <polybench.h>
#include "ludcmp.h"

#define GPU_DEVICE 0

/* 数组初始化：生成正定矩阵以确保 LU 分解稳定性 */
static void init_array (int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_1D(b,N,n), 
                        DATA_TYPE POLYBENCH_1D(x,N,n), DATA_TYPE POLYBENCH_1D(y,N,n))
{
    int i, j, r, s, t;
    DATA_TYPE fn = (DATA_TYPE)n;
    for (i = 0; i < n; i++) {
        x[i] = 0; y[i] = 0; b[i] = (i+1)/fn/2.0 + 4;
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++) A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
        for (j = i+1; j < n; j++) A[i][j] = 0;
        A[i][i] = 1;
    }
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
    for (r = 0; r < n; ++r) for (s = 0; s < n; ++s) (POLYBENCH_ARRAY(B))[r][s] = 0;
    for (t = 0; t < n; ++t) for (r = 0; r < n; ++r) for (s = 0; s < n; ++s)
        (POLYBENCH_ARRAY(B))[r][s] += A[r][t] * A[s][t];
    for (r = 0; r < n; ++r) for (s = 0; s < n; ++s) A[r][s] = (POLYBENCH_ARRAY(B))[r][s];
    POLYBENCH_FREE_ARRAY(B);
}

static void print_array_to_file(FILE *target, int n,
                                DATA_TYPE POLYBENCH_1D(x, N, n))
{
    int i;
    for (i = 0; i < n; i++) {
        fprintf(target, DATA_PRINTF_MODIFIER, x[i]);
        if ((i + 1) % 20 == 0) fprintf(target, "\n");
    }
}

/* 结果打印 */
static void print_array(int n, DATA_TYPE POLYBENCH_1D(x,N,n))
{
    int i;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("x");
    for (i = 0; i < n; i++) {
        if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
        fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
    }
    POLYBENCH_DUMP_END("x");
    POLYBENCH_DUMP_FINISH;
}

/* 内核 1：LU 分解部分 - 处理 A[i][j] 其中 j < i */
__global__ void ludcmp_kernel_lu_low(int n, DATA_TYPE *a, int i, int j)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double w = (double)a[i * n + j];
        for (int k = 0; k < j; k++) {
            w -= (double)a[i * n + k] * (double)a[k * n + j];
        }
        a[i * n + j] = (DATA_TYPE)(w / a[j * n + j]);
    }
}

/* 内核 2：LU 分解部分 - 处理 A[i][j] 其中 j >= i */
__global__ void ludcmp_kernel_lu_high(int n, DATA_TYPE *a, int i, int j)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double w = (double)a[i * n + j];
        for (int k = 0; k < i; k++) {
            w -= (double)a[i * n + k] * (double)a[k * n + j];
        }
        a[i * n + j] = (DATA_TYPE)w;
    }
}

/* 内核 3：前向替换计算 y[i] */
__global__ void ludcmp_kernel_forward(int n, DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *y, int i)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double w = (double)b[i];
        for (int j = 0; j < i; j++) {
            w -= (double)a[i * n + j] * (double)y[j];
        }
        y[i] = (DATA_TYPE)w;
    }
}

/* 内核 4：后向替换计算 x[i] */
__global__ void ludcmp_kernel_backward(int n, DATA_TYPE *a, DATA_TYPE *y, DATA_TYPE *x, int i)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double w = (double)y[i];
        for (int j = i + 1; j < n; j++) {
            w -= (double)a[i * n + j] * (double)x[j];
        }
        x[i] = (DATA_TYPE)(w / a[i * n + i]);
    }
}

void ludcmpCuda(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_1D(b,N,n), 
                DATA_TYPE POLYBENCH_1D(x,N,n), DATA_TYPE POLYBENCH_1D(y,N,n))
{
    DATA_TYPE *A_gpu, *b_gpu, *x_gpu, *y_gpu;
    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * N * N);
    cudaMalloc((void **)&b_gpu, sizeof(DATA_TYPE) * N);
    cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * N);
    cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * N);

    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);

    polybench_start_instruments;

    // 1. LU 分解阶段 (序列化控制)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            ludcmp_kernel_lu_low<<<1, 1>>>(n, A_gpu, i, j);
            cudaDeviceSynchronize();
        }
        for (int j = i; j < n; j++) {
            ludcmp_kernel_lu_high<<<1, 1>>>(n, A_gpu, i, j);
            cudaDeviceSynchronize();
        }
    }

    // 2. 前向替换 (A*y = b)
    for (int i = 0; i < n; i++) {
        ludcmp_kernel_forward<<<1, 1>>>(n, A_gpu, b_gpu, y_gpu, i);
        cudaDeviceSynchronize();
    }

    // 3. 后向替换 (U*x = y)
    for (int i = n - 1; i >= 0; i--) {
        ludcmp_kernel_backward<<<1, 1>>>(n, A_gpu, y_gpu, x_gpu, i);
        cudaDeviceSynchronize();
    }

    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(x, x_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);

    cudaFree(A_gpu); cudaFree(b_gpu); cudaFree(x_gpu); cudaFree(y_gpu);
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
    POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);

    init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y));
    cudaSetDevice(GPU_DEVICE);
    ludcmpCuda(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(b), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y));

    if (verbose)
    {
        FILE *output_file = fopen("c_ludcmp_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(x)));
            fclose(output_file);
        }
    }

    POLYBENCH_FREE_ARRAY(A); POLYBENCH_FREE_ARRAY(b);
    POLYBENCH_FREE_ARRAY(x); POLYBENCH_FREE_ARRAY(y);
    return 0;
}
