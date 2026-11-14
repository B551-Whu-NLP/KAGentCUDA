/**
 * trisolv.cu: CUDA implementation of PolyBench TRISOLV
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <string.h>

#define POLYBENCH_TIME 1
#include <polybench.h>
#include "trisolv.h"

#define GPU_DEVICE 0

/* 数组初始化：严格对应 PolyBench 2016 逻辑 */
static void init_array(int n,
                DATA_TYPE POLYBENCH_2D(L,N,N,n,n),
                DATA_TYPE POLYBENCH_1D(x,N,n),
                DATA_TYPE POLYBENCH_1D(b,N,n))
{
    int i, j;
    for (i = 0; i < n; i++) {
        x[i] = -999;
        b[i] = i;
        for (j = 0; j <= i; j++)
            L[i][j] = (DATA_TYPE) (i + n - j + 1) * 2 / n;
    }
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
        fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x[i]);
        if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    }
    POLYBENCH_DUMP_END("x");
    POLYBENCH_DUMP_FINISH;
}

/**
 * TRISOLV CUDA Kernel
 * 每一个线程负责计算一个 x[i]
 * 这种单元素更新方式能最大程度保证数值健壮性
 */
__global__ void trisolv_kernel(int n, DATA_TYPE *l, DATA_TYPE *x, DATA_TYPE *b, int i)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double tmp = (double)b[i];
        for (int j = 0; j < i; j++) {
            // 读取之前计算好的 x[j]
            tmp -= (double)l[i * n + j] * (double)x[j];
        }
        x[i] = (DATA_TYPE)(tmp / l[i * n + i]);
    }
}

void trisolvCuda(int n, DATA_TYPE POLYBENCH_2D(L,N,N,n,n), 
                 DATA_TYPE POLYBENCH_1D(x,N,n), 
                 DATA_TYPE POLYBENCH_1D(b,N,n))
{
    DATA_TYPE *L_gpu, *x_gpu, *b_gpu;

    // 分配显存
    cudaMalloc((void **)&L_gpu, sizeof(DATA_TYPE) * N * N);
    cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * N);
    cudaMalloc((void **)&b_gpu, sizeof(DATA_TYPE) * N);

    // 拷贝数据
    cudaMemcpy(L_gpu, L, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);

    polybench_start_instruments;

    // 序列化计算：每一个 i 依赖于 0 到 i-1 的结果
    for (int i = 0; i < n; i++) {
        trisolv_kernel<<<1, 1>>>(n, L_gpu, x_gpu, b_gpu, i);
        // 必须同步，因为下一轮循环需要读取 x[i]
        cudaDeviceSynchronize();
    }

    polybench_stop_instruments;
    polybench_print_instruments;

    // 拷贝结果回主机
    cudaMemcpy(x, x_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);

    // 释放资源
    cudaFree(L_gpu);
    cudaFree(x_gpu);
    cudaFree(b_gpu);
}

int main(int argc, char** argv)
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }

    int n = N;

    POLYBENCH_2D_ARRAY_DECL(L, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(b, DATA_TYPE, N, n);

    init_array(n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b));

    cudaSetDevice(GPU_DEVICE);
    trisolvCuda(n, POLYBENCH_ARRAY(L), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(b));

    if (verbose)
    {
        FILE *output_file = fopen("c_trisolv_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(x)));
            fclose(output_file);
        }
    }

    POLYBENCH_FREE_ARRAY(L);
    POLYBENCH_FREE_ARRAY(x);
    POLYBENCH_FREE_ARRAY(b);

    return 0;
}
