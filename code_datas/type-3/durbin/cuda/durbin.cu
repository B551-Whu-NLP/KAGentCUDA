/**
 * durbin.cu: CUDA implementation of Durbin algorithm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <string.h>

#define POLYBENCH_TIME 1
#include <polybench.h>
#include "durbin.h"

#define GPU_DEVICE 0

/* 数组初始化 */
static void init_array (int n, DATA_TYPE POLYBENCH_1D(r,N,n))
{
    int i;
    for (i = 0; i < n; i++) {
        r[i] = (DATA_TYPE)(n + 1 - i);
    }
}

static void print_array_to_file(FILE *target, int n,
                                DATA_TYPE POLYBENCH_1D(y, N, n))
{
    int i;
    for (i = 0; i < n; i++)
    {
        fprintf(target, DATA_PRINTF_MODIFIER, y[i]);
        if ((i + 1) % 20 == 0)
            fprintf(target, "\n");
    }
}

/* 结果打印 */
static void print_array(int n, DATA_TYPE POLYBENCH_1D(y,N,n))
{
    int i;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("y");
    for (i = 0; i < n; i++) {
        if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
        fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
    }
    POLYBENCH_DUMP_END("y");
    POLYBENCH_DUMP_FINISH;
}

/**
 * Kernel 1: 计算 sum 值
 * 由于每一轮 k 只需要一个 sum 值，且 sum 依赖于 y[0...k-1]
 */
__global__ void durbin_kernel_sum(int k, DATA_TYPE *r, DATA_TYPE *y, DATA_TYPE *sum_out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double local_sum = 0.0;
        for (int i = 0; i < k; i++) {
            local_sum += (double)r[k - i - 1] * (double)y[i];
        }
        *sum_out = (DATA_TYPE)local_sum;
    }
}

/**
 * Kernel 2: 更新向量 z 并同步回 y
 * 处理 z[i] = y[i] + alpha * y[k-i-1]
 */
__global__ void durbin_kernel_update(int k, DATA_TYPE alpha, DATA_TYPE *y, DATA_TYPE *z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) {
        z[i] = y[i] + alpha * y[k - i - 1];
    }
}

/**
 * Kernel 3: 将 z 拷贝回 y (也可在 Kernel 2 合并，但为对齐原算法逻辑分步执行)
 */
__global__ void durbin_kernel_copy(int k, DATA_TYPE *y, DATA_TYPE *z, DATA_TYPE alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) {
        y[i] = z[i];
    }
    if (i == 0) {
        y[k] = alpha;
    }
}

void durbinCuda(int n, DATA_TYPE *r, DATA_TYPE *y) {
    DATA_TYPE *r_gpu, *y_gpu, *z_gpu, *sum_gpu;
    cudaMalloc(&r_gpu, n * sizeof(DATA_TYPE));
    cudaMalloc(&y_gpu, n * sizeof(DATA_TYPE));
    cudaMalloc(&z_gpu, n * sizeof(DATA_TYPE));
    cudaMalloc(&sum_gpu, sizeof(DATA_TYPE));

    cudaMemcpy(r_gpu, r, n * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    
    // 初始步
    DATA_TYPE h_y0;
    cudaMemcpy(&h_y0, r_gpu, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    h_y0 = -h_y0;
    cudaMemcpy(y_gpu, &h_y0, sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    DATA_TYPE alpha = h_y0;
    DATA_TYPE beta = SCALAR_VAL(1.0);
    DATA_TYPE sum_val;

    polybench_start_instruments;

    for (int k = 1; k < n; k++) {
        // 1. 更新 beta
        beta = (1.0 - (double)alpha * alpha) * beta;

        // 2. 计算 sum
        durbin_kernel_sum<<<1, 1>>>(k, r_gpu, y_gpu, sum_gpu);
        cudaMemcpy(&sum_val, sum_gpu, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

        // 3. 计算新的 alpha
        DATA_TYPE r_k;
        cudaMemcpy(&r_k, r_gpu + k, sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
        alpha = -(r_k + sum_val) / beta;

        // 4. 更新向量
        dim3 block(256);
        dim3 grid((k + 255) / 256);
        durbin_kernel_update<<<grid, block>>>(k, alpha, y_gpu, z_gpu);
        durbin_kernel_copy<<<grid, block>>>(k, y_gpu, z_gpu, alpha);
    }

    cudaDeviceSynchronize();
    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(y, y_gpu, n * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    cudaFree(r_gpu); cudaFree(y_gpu); cudaFree(z_gpu); cudaFree(sum_gpu);
}

int main(int argc, char** argv)
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }

    int n = N;
    POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);

    init_array(n, POLYBENCH_ARRAY(r));

    cudaSetDevice(GPU_DEVICE);
    durbinCuda(n, POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(y));

    if (verbose)
    {
        FILE *output_file = fopen("c_durbin_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(y)));
            fclose(output_file);
        }
    }

    POLYBENCH_FREE_ARRAY(r);
    POLYBENCH_FREE_ARRAY(y);

    return 0;
}
