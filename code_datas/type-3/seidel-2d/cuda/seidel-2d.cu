/**
 * seidel-2d.cu: CUDA implementation of Seidel-2D stencil
 * 使用波前法（Wavefront）解决强数据依赖
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <string.h>

#define POLYBENCH_TIME 1
#include <polybench.h>
#include "seidel-2d.h"

#define GPU_DEVICE 0

/* 数组初始化：严格对应 PolyBench 2016 逻辑 */
static void init_array (int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
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

/* 结果打印 */
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

/**
 * Seidel-2D Wavefront Kernel
 * 每个 step 代表一条对角线 i + j = const
 */
__global__ void seidel_wavefront_kernel(int n, DATA_TYPE *A, int step)
{
    // 在第 step 步，我们要处理所有满足 i + j = step 的点
    // 我们让 threadIdx.x 映射为 i
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = step - i;

    // 边界检查：必须在矩阵内部 (1 到 n-2)
    if (i >= 1 && i <= n - 2 && j >= 1 && j <= n - 2)
    {
        A[i * n + j] = (A[(i-1) * n + (j-1)] + A[(i-1) * n + j] + A[(i-1) * n + (j+1)]
                      + A[i * n + (j-1)]     + A[i * n + j]     + A[i * n + (j+1)]
                      + A[(i+1) * n + (j-1)] + A[(i+1) * n + j] + A[(i+1) * n + (j+1)]) 
                      / SCALAR_VAL(9.0);
    }
}

void seidel2DCuda(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
    DATA_TYPE *A_gpu;
    size_t size = (size_t)n * n * sizeof(DATA_TYPE);

    cudaMalloc(&A_gpu, size);
    cudaMemcpy(A_gpu, A, size, cudaMemcpyHostToDevice);

    // 线程块大小
    int threadsPerBlock = 256;

    polybench_start_instruments;

    for (int t = 0; t < tsteps; t++)
    {
        // 波前扫描：对角线和从 2 (i=1, j=1) 到 2*n-4 (i=n-2, j=n-2)
        for (int step = 2; step <= 2 * n - 4; step++)
        {
            // 在当前对角线 step 下，i 的可能范围受限
            // 因为 j = step - i 且 j >= 1，所以 i <= step - 1
            // 又因为 i <= n - 2
            int i_max = (step - 1 < n - 2) ? step - 1 : n - 2;
            int i_min = (step - (n - 2) > 1) ? step - (n - 2) : 1;
            int num_threads = i_max - i_min + 1;

            if (num_threads > 0) {
                int grid = (num_threads + threadsPerBlock - 1) / threadsPerBlock;
                // 注意：这里需要根据 i_min 调整 kernel 内部的 i 映射，
                // 或者简单地让 kernel 处理所有满足 step 的 i
                seidel_wavefront_kernel<<<grid, threadsPerBlock>>>(n, A_gpu, step);
                
                // 这里每一波之后必须强制同步
                cudaDeviceSynchronize();
            }
        }
    }

    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(A, A_gpu, size, cudaMemcpyDeviceToHost);
    cudaFree(A_gpu);
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

    init_array(n, POLYBENCH_ARRAY(A));

    cudaSetDevice(GPU_DEVICE);
    seidel2DCuda(tsteps, n, POLYBENCH_ARRAY(A));

    if (verbose)
    {
        FILE *output_file = fopen("c_seidel2d_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(A)));
            fclose(output_file);
        }
    }

    POLYBENCH_FREE_ARRAY(A);

    return 0;
}
