/**
 * covariance.cu: 这是一个 CUDA 版本的 PolyBench covariance 算法实现。
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
#include "covariance.h"

// 定义错误阈值
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define GPU_DEVICE 0

/* 数组初始化 (与 C 代码逻辑一致) */
static
void init_array (int m, int n,
                 DATA_TYPE *float_n,
                 DATA_TYPE POLYBENCH_2D(data,N,M,n,m))
{
  int i, j;
  *float_n = (DATA_TYPE)n;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = ((DATA_TYPE) i*j) / M;
}

static
void print_array_to_file(FILE *target, int m,
                         DATA_TYPE POLYBENCH_2D(cov, M, M, m, m))
{
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < m; j++)
        {
            fprintf(target, DATA_PRINTF_MODIFIER, cov[i][j]);
            if ((i * m + j + 1) % 20 == 0)
                fprintf(target, "\n");
        }
}

/* 打印输出结果 */
static
void print_array(int m,
                 DATA_TYPE POLYBENCH_2D(cov,M,M,m,m))
{
  int i, j;
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("cov");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, cov[i][j]);
    }
  POLYBENCH_DUMP_END("cov");
  POLYBENCH_DUMP_FINISH;
}

/* GPU 核函数 1: 计算每列的均值 */
__global__ void mean_kernel(int m, int n, DATA_TYPE float_n, DATA_TYPE *mean, DATA_TYPE *data)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < m)
    {
        mean[j] = 0.0;
        int i;
        for (i = 0; i < n; i++)
        {
            mean[j] += data[i * m + j];
        }
        mean[j] /= float_n;
    }
}

/* GPU 核函数 2: 中心化数据 (减去均值) */
__global__ void reduce_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *data)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < m)
    {
        data[i * m + j] -= mean[j];
    }
}

/* GPU 核函数 3: 计算协方差矩阵 (M x M) */
__global__ void covar_kernel(int m, int n, DATA_TYPE float_n, DATA_TYPE *cov, DATA_TYPE *data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m)
    {
        int j, k;
        for (j = i; j < m; j++)
        {
            DATA_TYPE val = 0.0;
            for (k = 0; k < n; k++)
            {
                val += data[k * m + i] * data[k * m + j];
            }
            val /= (float_n - 1.0);
            cov[i * m + j] = val;
            cov[j * m + i] = val; // 对称性
        }
    }
}

/* CUDA 管理与执行 */
void covarianceCuda(int m, int n, DATA_TYPE float_n, 
                    DATA_TYPE POLYBENCH_2D(data, N, M, n, m), 
                    DATA_TYPE POLYBENCH_2D(cov, M, M, m, m), 
                    DATA_TYPE POLYBENCH_1D(mean, M, m))
{
    DATA_TYPE *data_gpu;
    DATA_TYPE *mean_gpu;
    DATA_TYPE *cov_gpu;

    // 分配显存
    cudaMalloc((void **)&data_gpu, sizeof(DATA_TYPE) * N * M);
    cudaMalloc((void **)&mean_gpu, sizeof(DATA_TYPE) * M);
    cudaMalloc((void **)&cov_gpu, sizeof(DATA_TYPE) * M * M);

    // 数据 HtoD
    cudaMemcpy(data_gpu, data, sizeof(DATA_TYPE) * N * M, cudaMemcpyHostToDevice);

    // 核函数配置
    dim3 block1(256, 1);
    dim3 grid1((m + block1.x - 1) / block1.x, 1);

    dim3 block2(32, 8);
    dim3 grid2((m + block2.x - 1) / block2.x, (n + block2.y - 1) / block2.y);

    dim3 block3(256, 1);
    dim3 grid3((m + block3.x - 1) / block3.x, 1);

    polybench_start_instruments;

    // 1. 计算均值
    mean_kernel<<<grid1, block1>>>(m, n, float_n, mean_gpu, data_gpu);
    cudaDeviceSynchronize();

    // 2. 中心化数据
    reduce_kernel<<<grid2, block2>>>(m, n, mean_gpu, data_gpu);
    cudaDeviceSynchronize();

    // 3. 计算协方差
    covar_kernel<<<grid3, block3>>>(m, n, float_n, cov_gpu, data_gpu);
    cudaDeviceSynchronize();

    polybench_stop_instruments;
    polybench_print_instruments;

    // 数据 DtoH
    cudaMemcpy(cov, cov_gpu, sizeof(DATA_TYPE) * M * M, cudaMemcpyDeviceToHost);

    // 释放显存
    cudaFree(data_gpu);
    cudaFree(mean_gpu);
    cudaFree(cov_gpu);
}

int main(int argc, char** argv)
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }

    int n = N;
    int m = M;
    DATA_TYPE float_n;

    /* 变量声明 */
    POLYBENCH_2D_ARRAY_DECL(data, DATA_TYPE, N, M, n, m);
    POLYBENCH_2D_ARRAY_DECL(cov, DATA_TYPE, M, M, m, m);
    POLYBENCH_1D_ARRAY_DECL(mean, DATA_TYPE, M, m);

    /* 初始化 */
    init_array(m, n, &float_n, POLYBENCH_ARRAY(data));

    /* 设置 GPU 设备 */
    cudaSetDevice(GPU_DEVICE);

    /* 执行 CUDA 计算 */
    covarianceCuda(m, n, float_n, 
                   POLYBENCH_ARRAY(data), 
                   POLYBENCH_ARRAY(cov), 
                   POLYBENCH_ARRAY(mean));

    /* 防止死代码消除 (DCE) */
    if (verbose)
    {
        FILE *output_file = fopen("c_covariance_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, m, POLYBENCH_ARRAY(cov)));
            fclose(output_file);
        }
    }

    /* 清理内存 */
    POLYBENCH_FREE_ARRAY(data);
    POLYBENCH_FREE_ARRAY(cov);
    POLYBENCH_FREE_ARRAY(mean);

    return 0;
}