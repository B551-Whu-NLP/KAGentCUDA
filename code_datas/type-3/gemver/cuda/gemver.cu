/**
 * gemver.cu: 这是一个 CUDA 版本的 PolyBench gemver 算法实现。
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
#include "gemver.h"

/* 定义 CUDA 线程块大小 */
#ifndef DIM_THREAD_BLOCK_X
# define DIM_THREAD_BLOCK_X 32
#endif
#ifndef DIM_THREAD_BLOCK_Y
# define DIM_THREAD_BLOCK_Y 8
#endif

#define GPU_DEVICE 0

/* 数组初始化 (保持与串行 C 代码逻辑一致) */
static
void init_array (int n,
		 DATA_TYPE *alpha,
		 DATA_TYPE *beta,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_1D(u1,N,n),
		 DATA_TYPE POLYBENCH_1D(v1,N,n),
		 DATA_TYPE POLYBENCH_1D(u2,N,n),
		 DATA_TYPE POLYBENCH_1D(v2,N,n),
		 DATA_TYPE POLYBENCH_1D(w,N,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n),
		 DATA_TYPE POLYBENCH_1D(y,N,n),
		 DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;

  DATA_TYPE fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
    {
      u1[i] = i;
      u2[i] = ((i+1)/fn)/2.0;
      v1[i] = ((i+1)/fn)/4.0;
      v2[i] = ((i+1)/fn)/6.0;
      y[i] = ((i+1)/fn)/8.0;
      z[i] = ((i+1)/fn)/9.0;
      x[i] = 0.0;
      w[i] = 0.0;
      for (j = 0; j < n; j++)
        A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }
}

static
void print_array_to_file(FILE *target, int n,
                         DATA_TYPE POLYBENCH_1D(w, N, n))
{
    int i;
    for (i = 0; i < n; i++)
    {
        if (i % 20 == 0)
            fprintf(target, "\n");
        fprintf(target, DATA_PRINTF_MODIFIER, w[i]);
    }
}

/* 打印输出结果 (用于校验或防止 DCE) */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(w,N,n))
{
  int i;
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("w");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, w[i]);
  }
  POLYBENCH_DUMP_END("w");
  POLYBENCH_DUMP_FINISH;
}

__global__ void gemver_kernel1(int n, DATA_TYPE *A, DATA_TYPE *u1, DATA_TYPE *v1, DATA_TYPE *u2, DATA_TYPE *v2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n) {
        // 对应 A[i][j] = A[i][j] + u1[i]*v1[j] + u2[i]*v2[j]
        A[i * n + j] += u1[i] * v1[j] + u2[i] * v2[j];
    }
}

// 2. x = x + beta * A' * y + z (合并了串行中的两个循环)
__global__ void gemver_kernel2(int n, DATA_TYPE beta, DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        DATA_TYPE dot = SCALAR_VAL(0.0);
        for (int j = 0; j < n; j++) {
            // A[j][i] 是列访问，对应 A' * y
            dot += beta * A[j * n + i] * y[j];
        }
        // 显式赋值：x[i] = 初始x(0) + dot + z[i]
        x[i] = x[i] + dot + z[i];
    }
}

// 3. w = w + alpha * A * x
__global__ void gemver_kernel3(int n, DATA_TYPE alpha, DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        DATA_TYPE dot = SCALAR_VAL(0.0);
        for (int j = 0; j < n; j++) {
            // A[i][j] 是行访问
            dot += alpha * A[i * n + j] * x[j];
        }
        // 显式赋值
        w[i] = w[i] + dot;
    }
}

/* CUDA 管理函数 */
void gemverCuda(int n, DATA_TYPE alpha, DATA_TYPE beta,
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(u1,N,n),
		DATA_TYPE POLYBENCH_1D(v1,N,n),
		DATA_TYPE POLYBENCH_1D(u2,N,n),
		DATA_TYPE POLYBENCH_1D(v2,N,n),
		DATA_TYPE POLYBENCH_1D(w,N,n),
		DATA_TYPE POLYBENCH_1D(x,N,n),
		DATA_TYPE POLYBENCH_1D(y,N,n),
		DATA_TYPE POLYBENCH_1D(z,N,n))
{
	DATA_TYPE *A_gpu, *x_gpu, *y_gpu, *z_gpu, *v1_gpu, *v2_gpu, *u1_gpu, *u2_gpu, *w_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * N * N);
	cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&z_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&w_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&v1_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&v2_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&u1_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&u2_gpu, sizeof(DATA_TYPE) * N);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(z_gpu, z, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(w_gpu, w, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(v1_gpu, v1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(v2_gpu, v2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(u1_gpu, u1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(u2_gpu, u2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
    polybench_start_instruments;

	dim3 block2D(32, 8);
    dim3 grid2D((n + block2D.x - 1) / block2D.x, (n + block2D.y - 1) / block2D.y);

    dim3 block1D(256);
    dim3 grid1D((n + block1D.x - 1) / block1D.x);

    // 严格顺序执行
    gemver_kernel1<<<grid2D, block2D>>>(n, A_gpu, u1_gpu, v1_gpu, u2_gpu, v2_gpu);
    cudaDeviceSynchronize();

    gemver_kernel2<<<grid1D, block1D>>>(n, beta, A_gpu, x_gpu, y_gpu, z_gpu);
    cudaDeviceSynchronize();

    gemver_kernel3<<<grid1D, block1D>>>(n, alpha, A_gpu, x_gpu, w_gpu);
    cudaDeviceSynchronize();

	/* 停止计时 */
	polybench_stop_instruments;
	polybench_print_instruments;

	cudaMemcpy(w, w_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);
	
	cudaFree(A_gpu); cudaFree(x_gpu); cudaFree(y_gpu); cudaFree(z_gpu);
	cudaFree(w_gpu); cudaFree(v1_gpu); cudaFree(v2_gpu); cudaFree(u1_gpu); cudaFree(u2_gpu);
}

int main(int argc, char** argv)
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }

    int n = N;
    DATA_TYPE alpha, beta;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);

    init_array (n, &alpha, &beta,
	      POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(v1),
	      POLYBENCH_ARRAY(u2), POLYBENCH_ARRAY(v2), POLYBENCH_ARRAY(w),
	      POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z));

    cudaSetDevice(GPU_DEVICE);

    gemverCuda(n, alpha, beta,
	     POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(v1),
	     POLYBENCH_ARRAY(u2), POLYBENCH_ARRAY(v2), POLYBENCH_ARRAY(w),
	     POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z));

    if (verbose)
    {
        FILE *output_file = fopen("c_gemver_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(w)));
            fclose(output_file);
        }
    }

    POLYBENCH_FREE_ARRAY(A); POLYBENCH_FREE_ARRAY(u1); POLYBENCH_FREE_ARRAY(v1);
    POLYBENCH_FREE_ARRAY(u2); POLYBENCH_FREE_ARRAY(v2); POLYBENCH_FREE_ARRAY(w);
    POLYBENCH_FREE_ARRAY(x); POLYBENCH_FREE_ARRAY(y); POLYBENCH_FREE_ARRAY(z);

    return 0;
}