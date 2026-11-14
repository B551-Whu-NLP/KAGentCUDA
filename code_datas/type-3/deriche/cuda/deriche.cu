/**
 * deriche.cu: CUDA implementation of Deriche edge detector
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <string.h>

#define POLYBENCH_TIME 1
#include <polybench.h>
#include "deriche.h"

#define GPU_DEVICE 0

/* 数组初始化：严格对应 2016 C 串行逻辑 */
static void init_array(int w, int h, DATA_TYPE* alpha,
                DATA_TYPE POLYBENCH_2D(imgIn,W,H,w,h),
                DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h))
{
    int i, j;
    *alpha = 0.25;
    for (i = 0; i < w; i++)
        for (j = 0; j < h; j++)
            imgIn[i][j] = (DATA_TYPE) ((313*i+991*j)%65536) / 65535.0f;
}

static void print_array_to_file(FILE *target, int w, int h,
                                DATA_TYPE POLYBENCH_2D(imgOut, W, H, w, h))
{
    int i, j;
    for (i = 0; i < w; i++)
        for (j = 0; j < h; j++)
        {
            fprintf(target, DATA_PRINTF_MODIFIER, imgOut[i][j]);
            if ((i * h + j + 1) % 20 == 0)
                fprintf(target, "\n");
        }
}

/* 结果打印 */
static void print_array(int w, int h, DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h))
{
    int i, j;
    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("imgOut");
    for (i = 0; i < w; i++)
        for (j = 0; j < h; j++) {
            if ((i * h + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
            fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, imgOut[i][j]);
        }
    POLYBENCH_DUMP_END("imgOut");
    POLYBENCH_DUMP_FINISH;
}

/* ------------------- Kernels ------------------- */

// 1. 水平正向扫描 (Forward Horizontal)
__global__ void deriche_horiz_fwd_kernel(int w, int h, DATA_TYPE a1, DATA_TYPE a2, DATA_TYPE b1, DATA_TYPE b2, 
                                        DATA_TYPE *imgIn, DATA_TYPE *y1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w) {
        DATA_TYPE xm1 = 0, ym1 = 0, ym2 = 0;
        for (int j = 0; j < h; j++) {
            y1[i * h + j] = a1 * imgIn[i * h + j] + a2 * xm1 + b1 * ym1 + b2 * ym2;
            xm1 = imgIn[i * h + j];
            ym2 = ym1;
            ym1 = y1[i * h + j];
        }
    }
}

// 2. 水平反向扫描 (Backward Horizontal)
__global__ void deriche_horiz_bwd_kernel(int w, int h, DATA_TYPE a3, DATA_TYPE a4, DATA_TYPE b1, DATA_TYPE b2, 
                                        DATA_TYPE *imgIn, DATA_TYPE *y2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w) {
        DATA_TYPE xp1 = 0, xp2 = 0, yp1 = 0, yp2 = 0;
        for (int j = h - 1; j >= 0; j--) {
            y2[i * h + j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
            xp2 = xp1;
            xp1 = imgIn[i * h + j];
            yp2 = yp1;
            yp1 = y2[i * h + j];
        }
    }
}

// 3. 中间/最终合并与赋值 (Merge and Transpose-Ready)
__global__ void deriche_merge_kernel(int w, int h, DATA_TYPE c, DATA_TYPE *y1, DATA_TYPE *y2, DATA_TYPE *imgOut) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w && j < h) {
        imgOut[i * h + j] = c * (y1[i * h + j] + y2[i * h + j]);
    }
}

// 4. 垂直正向扫描 (Forward Vertical)
__global__ void deriche_vert_fwd_kernel(int w, int h, DATA_TYPE a5, DATA_TYPE a6, DATA_TYPE b1, DATA_TYPE b2, 
                                       DATA_TYPE *imgOut, DATA_TYPE *y1) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < h) {
        DATA_TYPE tm1 = 0, ym1 = 0, ym2 = 0;
        for (int i = 0; i < w; i++) {
            y1[i * h + j] = a5 * imgOut[i * h + j] + a6 * tm1 + b1 * ym1 + b2 * ym2;
            tm1 = imgOut[i * h + j];
            ym2 = ym1;
            ym1 = y1[i * h + j];
        }
    }
}

// 5. 垂直反向扫描 (Backward Vertical)
__global__ void deriche_vert_bwd_kernel(int w, int h, DATA_TYPE a7, DATA_TYPE a8, DATA_TYPE b1, DATA_TYPE b2, 
                                       DATA_TYPE *imgOut, DATA_TYPE *y2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < h) {
        DATA_TYPE tp1 = 0, tp2 = 0, yp1 = 0, yp2 = 0;
        for (int i = w - 1; i >= 0; i--) {
            y2[i * h + j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
            tp2 = tp1;
            tp1 = imgOut[i * h + j];
            yp2 = yp1;
            yp1 = y2[i * h + j];
        }
    }
}

void dericheCuda(int w, int h, DATA_TYPE alpha, 
                DATA_TYPE *imgIn, DATA_TYPE *imgOut, DATA_TYPE *y1, DATA_TYPE *y2) {
    DATA_TYPE *imgIn_gpu, *imgOut_gpu, *y1_gpu, *y2_gpu;
    cudaMalloc(&imgIn_gpu, w * h * sizeof(DATA_TYPE));
    cudaMalloc(&imgOut_gpu, w * h * sizeof(DATA_TYPE));
    cudaMalloc(&y1_gpu, w * h * sizeof(DATA_TYPE));
    cudaMalloc(&y2_gpu, w * h * sizeof(DATA_TYPE));

    cudaMemcpy(imgIn_gpu, imgIn, w * h * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    // 计算参数
    DATA_TYPE k = (1.0-exp(-alpha))*(1.0-exp(-alpha))/(1.0+2.0*alpha*exp(-alpha)-exp(2.0*alpha));
    DATA_TYPE a1 = k, a5 = k;
    DATA_TYPE a2 = k*exp(-alpha)*(alpha-1.0), a6 = a2;
    DATA_TYPE a3 = k*exp(-alpha)*(alpha+1.0), a7 = a3;
    DATA_TYPE a4 = -k*exp(-2.0*alpha), a8 = a4;
    DATA_TYPE b1 = pow(2.0,-alpha);
    DATA_TYPE b2 = -exp(-2.0*alpha);
    DATA_TYPE c1 = 1, c2 = 1;

    dim3 block1D(256);
    dim3 gridW((w + 255) / 256);
    dim3 gridH((h + 255) / 256);
    dim3 block2D(32, 8);
    dim3 grid2D((h + 31) / 32, (w + 7) / 8);

    polybench_start_instruments;

    deriche_horiz_fwd_kernel<<<gridW, block1D>>>(w, h, a1, a2, b1, b2, imgIn_gpu, y1_gpu);
    deriche_horiz_bwd_kernel<<<gridW, block1D>>>(w, h, a3, a4, b1, b2, imgIn_gpu, y2_gpu);
    cudaDeviceSynchronize();

    deriche_merge_kernel<<<grid2D, block2D>>>(w, h, c1, y1_gpu, y2_gpu, imgOut_gpu);
    cudaDeviceSynchronize();

    deriche_vert_fwd_kernel<<<gridH, block1D>>>(w, h, a5, a6, b1, b2, imgOut_gpu, y1_gpu);
    deriche_vert_bwd_kernel<<<gridH, block1D>>>(w, h, a7, a8, b1, b2, imgOut_gpu, y2_gpu);
    cudaDeviceSynchronize();

    deriche_merge_kernel<<<grid2D, block2D>>>(w, h, c2, y1_gpu, y2_gpu, imgOut_gpu);
    cudaDeviceSynchronize();

    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(imgOut, imgOut_gpu, w * h * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    cudaFree(imgIn_gpu); cudaFree(imgOut_gpu); cudaFree(y1_gpu); cudaFree(y2_gpu);
}

int main(int argc, char** argv) {
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }

    int w = W; int h = H;
    DATA_TYPE alpha;
    POLYBENCH_2D_ARRAY_DECL(imgIn, DATA_TYPE, W, H, w, h);
    POLYBENCH_2D_ARRAY_DECL(imgOut, DATA_TYPE, W, H, w, h);
    POLYBENCH_2D_ARRAY_DECL(y1, DATA_TYPE, W, H, w, h);
    POLYBENCH_2D_ARRAY_DECL(y2, DATA_TYPE, W, H, w, h);

    init_array(w, h, &alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut));
    cudaSetDevice(GPU_DEVICE);
    dericheCuda(w, h, alpha, (DATA_TYPE*)imgIn, (DATA_TYPE*)imgOut, (DATA_TYPE*)y1, (DATA_TYPE*)y2);

    if (verbose)
    {
        FILE *output_file = fopen("c_deriche_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, w, h, POLYBENCH_ARRAY(imgOut)));
            fclose(output_file);
        }
    }

    POLYBENCH_FREE_ARRAY(imgIn); POLYBENCH_FREE_ARRAY(imgOut);
    POLYBENCH_FREE_ARRAY(y1); POLYBENCH_FREE_ARRAY(y2);
    return 0;
}