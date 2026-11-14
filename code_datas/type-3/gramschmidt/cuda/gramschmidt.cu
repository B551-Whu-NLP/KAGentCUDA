/**
 * gramschmidt.cu: CUDA implementation of PolyBench gramschmidt
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <string.h>

#define POLYBENCH_TIME 1
#include <polybench.h>
#include "gramschmidt.h"

#ifndef DIM_THREAD_BLOCK_X
# define DIM_THREAD_BLOCK_X 256
#endif

#define GPU_DEVICE 0

/* 数组初始化：严格对应 2016 C 串行逻辑 */
static void init_array(int m, int n,
                DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
                DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
                DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++) {
            A[i][j] = (((DATA_TYPE) ((i*j) % m) / m )*100) + 10;
            Q[i][j] = SCALAR_VAL(0.0);
        }
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            R[i][j] = SCALAR_VAL(0.0);
}

static void print_array_to_file(FILE *target, int m, int n,
                                DATA_TYPE POLYBENCH_2D(R, N, N, n, n),
                                DATA_TYPE POLYBENCH_2D(Q, M, N, m, n))
{
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            fprintf(target, DATA_PRINTF_MODIFIER, R[i][j]);
            if ((i * n + j + 1) % 20 == 0) fprintf(target, "\n");
        }
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++) {
            fprintf(target, DATA_PRINTF_MODIFIER, Q[i][j]);
            if ((i * n + j + 1) % 20 == 0) fprintf(target, "\n");
        }
}

/* 结果打印 */
static void print_array(int m, int n,
                 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
                 DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
                 DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
    int i, j;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("R");
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
            fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, R[i][j]);
        }
    POLYBENCH_DUMP_END("R");
    POLYBENCH_DUMP_BEGIN("Q");
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++) {
            if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
            fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, Q[i][j]);
        }
    POLYBENCH_DUMP_END("Q");
    POLYBENCH_DUMP_FINISH;
}

/* Kernel 1: 计算当前列 k 的范数 R[k][k] */
__global__ void gramschmidt_kernel1(int m, int n, DATA_TYPE *a, DATA_TYPE *r, int k)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        DATA_TYPE nrm = SCALAR_VAL(0.0);
        for (int i = 0; i < m; i++) {
            nrm += a[i * n + k] * a[i * n + k];
        }
        r[k * n + k] = SQRT_FUN(nrm);
    }
}

/* Kernel 2: 生成正交向量列 Q[i][k] */
__global__ void gramschmidt_kernel2(int m, int n, DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        DATA_TYPE r_kk = r[k * n + k];
        // 增加 epsilon 保护，防止除法导致的数值崩溃
        if (abs(r_kk) > 1e-20) {
            q[i * n + k] = a[i * n + k] / r_kk;
        } else {
            q[i * n + k] = 0.0;
        }
    }
}

/* Kernel 3: 更新后续列 (按列并行) */
/* 修复后的 Kernel 3 */
/* 修复后的 Kernel 3 */
/* 定义共享内存大小，需与 block 维度一致 */
#define DIM_THREAD_BLOCK_X 256

/* 优化后的 Kernel 3：基于线程块协作的规约更新 */
__global__ void gramschmidt_kernel3(int m, int n, DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > k && j < n) {
        // 关键：使用 double 存储中间累加值，哪怕 DATA_TYPE 是 float
        double tmp_r = 0.0; 
        
        for (int i = 0; i < m; i++) {
            tmp_r += (double)q[i * n + k] * (double)a[i * n + j];
        }
        
        r[k * n + j] = (DATA_TYPE)tmp_r;

        for (int i = 0; i < m; i++) {
            // 更新时也保持高精度中间计算
            a[i * n + j] = (DATA_TYPE)((double)a[i * n + j] - (double)q[i * n + k] * tmp_r);
        }
    }
}

void gramschmidtCuda(int m, int n, 
                    DATA_TYPE POLYBENCH_2D(A,M,N,m,n), 
                    DATA_TYPE POLYBENCH_2D(R,N,N,n,n), 
                    DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
    DATA_TYPE *a_gpu, *r_gpu, *q_gpu;

    // 分配显存
    cudaMalloc((void **)&a_gpu, sizeof(DATA_TYPE) * M * N);
    cudaMalloc((void **)&r_gpu, sizeof(DATA_TYPE) * N * N);
    cudaMalloc((void **)&q_gpu, sizeof(DATA_TYPE) * M * N);

    // 拷贝初始 A
    cudaMemcpy(a_gpu, A, sizeof(DATA_TYPE) * M * N, cudaMemcpyHostToDevice);
    // 初始化显存中的 R, Q 为 0 (重要：防止残留值影响累加)
    cudaMemset(r_gpu, 0, sizeof(DATA_TYPE) * N * N);
    cudaMemset(q_gpu, 0, sizeof(DATA_TYPE) * M * N);

    dim3 block(256);
    // grid 必须覆盖所有的列 n
    dim3 grid_n((n + block.x - 1) / block.x); 
    dim3 grid_m((m + block.x - 1) / block.x);
    
    polybench_start_instruments;

    for (int k = 0; k < n; k++) {
        gramschmidt_kernel1<<<1, 1>>>(m, n, a_gpu, r_gpu, k);
        cudaDeviceSynchronize();
        
        gramschmidt_kernel2<<<grid_m, block>>>(m, n, a_gpu, r_gpu, q_gpu, k);
        cudaDeviceSynchronize();

        // 重点：这里的 grid 必须基于 n (列数)
        gramschmidt_kernel3<<<grid_n, block>>>(m, n, a_gpu, r_gpu, q_gpu, k);
        cudaDeviceSynchronize();
    }

    polybench_stop_instruments;
    polybench_print_instruments;

    // 拷贝结果回主机
    cudaMemcpy(A, a_gpu, sizeof(DATA_TYPE) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(R, r_gpu, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(Q, q_gpu, sizeof(DATA_TYPE) * M * N, cudaMemcpyDeviceToHost);

    cudaFree(a_gpu); cudaFree(r_gpu); cudaFree(q_gpu);
}

int main(int argc, char** argv)
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }

    int m = M; int n = N;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, N, m, n);
    POLYBENCH_2D_ARRAY_DECL(R, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(Q, DATA_TYPE, M, N, m, n);

    init_array(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q));

    cudaSetDevice(GPU_DEVICE);
    gramschmidtCuda(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q));

    if (verbose)
    {
        FILE *output_file = fopen("c_gramschmidt_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, m, n, POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q)));
            fclose(output_file);
        }
    }

    POLYBENCH_FREE_ARRAY(A); POLYBENCH_FREE_ARRAY(R); POLYBENCH_FREE_ARRAY(Q);

    return 0;
}