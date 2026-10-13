/**
 * correlation.cu: 这是一个 CUDA 版本的 PolyBench correlation 算法实现。
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda.h>
#include <string.h>

#define POLYBENCH_TIME 1

#include <polybench.h>
#include "correlation.h"

// 定义错误阈值
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define GPU_DEVICE 0

/* 数组初始化 (与 C 代码逻辑一致) */
static
void init_array (int m,
                 int n,
                 DATA_TYPE *float_n,
                 DATA_TYPE POLYBENCH_2D(data,N,M,n,m))
{
  int i, j;
  *float_n = (DATA_TYPE)N;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = (DATA_TYPE)(i*j)/M + i;
}

/* GPU 核函数 1: 计算均值 */
__global__ void mean_kernel(int m, int n, DATA_TYPE float_n, DATA_TYPE *mean, DATA_TYPE *data)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < m)
    {
        mean[j] = 0.0;
        int i;
        for(i = 0; i < n; i++)
        {
            mean[j] += data[i * m + j];
        }
        mean[j] /= float_n;
    }
}

/* GPU 核函数 2: 计算标准差 */
__global__ void std_kernel(int m, int n, DATA_TYPE float_n, DATA_TYPE *mean, DATA_TYPE *std, DATA_TYPE *data)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    DATA_TYPE eps = 0.1;

    if (j < m)
    {
        std[j] = 0.0;
        int i;
        for(i = 0; i < n; i++)
        {
            std[j] += (data[i * m + j] - mean[j]) * (data[i * m + j] - mean[j]);
        }
        std[j] /= float_n;
        std[j] = sqrt(std[j]);
        if(std[j] <= eps) 
        {
            std[j] = 1.0;
        }
    }
}

/* GPU 核函数 3: 中心化与规约 */
__global__ void reduce_kernel(int m, int n, DATA_TYPE float_n, DATA_TYPE *mean, DATA_TYPE *std, DATA_TYPE *data)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < n && j < m)
    {
        data[i * m + j] -= mean[j];
        data[i * m + j] /= (sqrt(float_n) * std[j]);
    }
}

/* GPU 核函数 4: 计算相关矩阵 */
__global__ void corr_kernel(int m, int n, DATA_TYPE *corr, DATA_TYPE *data)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (m - 1))
    {
        corr[i * m + i] = 1.0;
        int j, k;
        for (j = i + 1; j < m; j++)
        {
            DATA_TYPE val = 0.0;
            for(k = 0; k < n; k++)
            {
                val += data[k * m + i] * data[k * m + j];
            }
            corr[i * m + j] = val;
            corr[j * m + i] = val;
        }
    }
    // 处理最后一个对角线元素
    if (i == m - 1) {
        corr[i * m + i] = 1.0;
    }
}

/* GPU 显存初始化与计算调度 */
void correlationCuda(int m, int n, DATA_TYPE float_n, DATA_TYPE POLYBENCH_2D(data, N, M, n, m), 
                     DATA_TYPE POLYBENCH_2D(corr, M, M, m, m), DATA_TYPE POLYBENCH_1D(mean, M, m), 
                     DATA_TYPE POLYBENCH_1D(stddev, M, m))
{
    DATA_TYPE *data_gpu;
    DATA_TYPE *stddev_gpu;
    DATA_TYPE *mean_gpu;
    DATA_TYPE *corr_gpu;

    cudaMalloc((void **)&data_gpu, sizeof(DATA_TYPE) * N * M);
    cudaMalloc((void **)&corr_gpu, sizeof(DATA_TYPE) * M * M);
    cudaMalloc((void **)&stddev_gpu, sizeof(DATA_TYPE) * M);
    cudaMalloc((void **)&mean_gpu, sizeof(DATA_TYPE) * M);

    cudaMemcpy(data_gpu, data, sizeof(DATA_TYPE) * N * M, cudaMemcpyHostToDevice);

    // 核函数配置
    dim3 block1(256, 1);
    dim3 grid1((m + block1.x - 1) / block1.x, 1);

    dim3 block3(32, 8);
    dim3 grid3((m + block3.x - 1) / block3.x, (n + block3.y - 1) / block3.y);

    dim3 block4(256, 1);
    dim3 grid4((m + block4.x - 1) / block4.x, 1);

    polybench_start_instruments;

    mean_kernel<<< grid1, block1 >>>(m, n, float_n, mean_gpu, data_gpu);
    cudaDeviceSynchronize();
    
    std_kernel<<< grid1, block1 >>>(m, n, float_n, mean_gpu, stddev_gpu, data_gpu);
    cudaDeviceSynchronize();
    
    reduce_kernel<<< grid3, block3 >>>(m, n, float_n, mean_gpu, stddev_gpu, data_gpu);
    cudaDeviceSynchronize();
    
    corr_kernel<<< grid4, block4 >>>(m, n, corr_gpu, data_gpu);
    cudaDeviceSynchronize();

    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(corr, corr_gpu, sizeof(DATA_TYPE) * M * M, cudaMemcpyDeviceToHost);
    
    cudaFree(data_gpu);
    cudaFree(corr_gpu);
    cudaFree(stddev_gpu);
    cudaFree(mean_gpu);
}

static
void print_array_to_file(FILE *target, int m,
                         DATA_TYPE POLYBENCH_2D(corr, M, M, m, m))
{
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < m; j++)
        {
            fprintf(target, DATA_PRINTF_MODIFIER, corr[i][j]);
            if ((i * m + j + 1) % 20 == 0)
                fprintf(target, "\n");
        }
}

/* 打印输出结果 */
static
void print_array(int m,
                 DATA_TYPE POLYBENCH_2D(corr,M,M,m,m))
{
  int i, j;
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("corr");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, corr[i][j]);
    }
  POLYBENCH_DUMP_END("corr");
  POLYBENCH_DUMP_FINISH;
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

    POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,N,M,n,m);
    POLYBENCH_2D_ARRAY_DECL(corr,DATA_TYPE,M,M,m,m);
    POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
    POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE,M,m);
    
    init_array(m, n, &float_n, POLYBENCH_ARRAY(data));
    
    // 设置设备
    cudaSetDevice(GPU_DEVICE);

    correlationCuda(m, n, float_n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(corr), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(stddev));

    if (verbose)
    {
        FILE *output_file = fopen("c_correlation_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, m, POLYBENCH_ARRAY(corr)));
            fclose(output_file);
        }
    }

    POLYBENCH_FREE_ARRAY(data);
    POLYBENCH_FREE_ARRAY(corr);
    POLYBENCH_FREE_ARRAY(mean);
    POLYBENCH_FREE_ARRAY(stddev);

    return 0;
}