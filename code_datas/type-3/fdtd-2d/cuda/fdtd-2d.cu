/**
 * fdtd2d.cu: CUDA implementation of PolyBench FDTD 2D
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
#include "fdtd-2d.h"

/* 定义线程块大小 */
#ifndef DIM_THREAD_BLOCK_X
# define DIM_THREAD_BLOCK_X 32
#endif
#ifndef DIM_THREAD_BLOCK_Y
# define DIM_THREAD_BLOCK_Y 8
#endif

#define GPU_DEVICE 0

/* 数组初始化：严格对应 2016 C 串行逻辑 */
static void init_array (int tmax, int nx, int ny, 
                        DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
                        DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
                        DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
                        DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
    int i, j;
    for (i = 0; i < tmax; i++)
        _fict_[i] = (DATA_TYPE) i;
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++) {
            ex[i][j] = ((DATA_TYPE) i*(j+1)) / nx;
            ey[i][j] = ((DATA_TYPE) i*(j+2)) / ny;
            hz[i][j] = ((DATA_TYPE) i*(j+3)) / nx;
        }
}

static
void print_array_to_file(FILE *target, int nx, int ny,
                         DATA_TYPE POLYBENCH_2D(ex, NX, NY, nx, ny),
                         DATA_TYPE POLYBENCH_2D(ey, NX, NY, nx, ny),
                         DATA_TYPE POLYBENCH_2D(hz, NX, NY, nx, ny))
{
    int i, j;
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++)
        {
            fprintf(target, DATA_PRINTF_MODIFIER, ex[i][j]);
            if ((i * nx + j + 1) % 20 == 0)
                fprintf(target, "\n");
        }
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++)
        {
            fprintf(target, DATA_PRINTF_MODIFIER, ey[i][j]);
            if ((i * nx + j + 1) % 20 == 0)
                fprintf(target, "\n");
        }
    for (i = 0; i < nx; i++)
        for (j = 0; j < ny; j++)
        {
            fprintf(target, DATA_PRINTF_MODIFIER, hz[i][j]);
            if ((i * nx + j + 1) % 20 == 0)
                fprintf(target, "\n");
        }
}

/* 结果打印：对应 PolyBench 2016 DUMP 规范 */
static
void print_array(int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("ex");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ex[i][j]);
    }
  POLYBENCH_DUMP_END("ex");
  POLYBENCH_DUMP_FINISH;

  POLYBENCH_DUMP_BEGIN("ey");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ey[i][j]);
    }
  POLYBENCH_DUMP_END("ey");

  POLYBENCH_DUMP_BEGIN("hz");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, hz[i][j]);
    }
  POLYBENCH_DUMP_END("hz");
}


/* Kernel 1: 更新 Ey 场 (包含源项注入) */
__global__ void fdtd_kernel1(int nx, int ny, DATA_TYPE* _fict_, DATA_TYPE *ey, DATA_TYPE *hz, int t)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny)
    {
        if (i == 0) 
            ey[i * ny + j] = _fict_[t];
        else
            ey[i * ny + j] -= 0.5f * (hz[i * ny + j] - hz[(i-1) * ny + j]);
    }
}

/* Kernel 2: 更新 Ex 场 */
__global__ void fdtd_kernel2(int nx, int ny, DATA_TYPE *ex, DATA_TYPE *hz)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < nx && j < ny && j > 0)
    {
        ex[i * ny + j] -= 0.5f * (hz[i * ny + j] - hz[i * ny + (j-1)]);
    }
}

/* Kernel 3: 更新 Hz 场 */
__global__ void fdtd_kernel3(int nx, int ny, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < (nx - 1) && j < (ny - 1))
    {   
        hz[i * ny + j] -= 0.7f * (ex[i * ny + (j + 1)] - ex[i * ny + j] + 
                                  ey[(i + 1) * ny + j] - ey[i * ny + j]);
    }
}

void fdtdCuda(int tmax, int nx, int ny, DATA_TYPE POLYBENCH_1D(_fict_, TMAX, TMAX), 
              DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny), 
              DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
    DATA_TYPE *_fict_gpu, *ex_gpu, *ey_gpu, *hz_gpu;

    cudaMalloc((void **)&_fict_gpu, sizeof(DATA_TYPE) * TMAX);
    cudaMalloc((void **)&ex_gpu, sizeof(DATA_TYPE) * NX * NY);
    cudaMalloc((void **)&ey_gpu, sizeof(DATA_TYPE) * NX * NY);
    cudaMalloc((void **)&hz_gpu, sizeof(DATA_TYPE) * NX * NY);

    cudaMemcpy(_fict_gpu, _fict_, sizeof(DATA_TYPE) * TMAX, cudaMemcpyHostToDevice);
    cudaMemcpy(ex_gpu, ex, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);
    cudaMemcpy(ey_gpu, ey, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);
    cudaMemcpy(hz_gpu, hz, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((size_t)ceil(((float)NY) / ((float)block.x)), (size_t)ceil(((float)NX) / ((float)block.y)));

    polybench_start_instruments;

    for(int t = 0; t < tmax; t++)
    {
        fdtd_kernel1<<<grid, block>>>(nx, ny, _fict_gpu, ey_gpu, hz_gpu, t);
        fdtd_kernel2<<<grid, block>>>(nx, ny, ex_gpu, hz_gpu);
        fdtd_kernel3<<<grid, block>>>(nx, ny, ex_gpu, ey_gpu, hz_gpu);
    }
    
    cudaDeviceSynchronize();

    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(ex, ex_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyDeviceToHost);
    cudaMemcpy(ey, ey_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyDeviceToHost);
    cudaMemcpy(hz, hz_gpu, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyDeviceToHost);
    
    cudaFree(_fict_gpu); cudaFree(ex_gpu); cudaFree(ey_gpu); cudaFree(hz_gpu);
}

int main(int argc, char** argv)
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0)
    {
        verbose = 1;
    }

    int tmax = TMAX;
    int nx = NX;
    int ny = NY;

    POLYBENCH_2D_ARRAY_DECL(ex, DATA_TYPE, NX, NY, nx, ny);
    POLYBENCH_2D_ARRAY_DECL(ey, DATA_TYPE, NX, NY, nx, ny);
    POLYBENCH_2D_ARRAY_DECL(hz, DATA_TYPE, NX, NY, nx, ny);
    POLYBENCH_1D_ARRAY_DECL(_fict_, DATA_TYPE, TMAX, tmax);

    init_array(tmax, nx, ny, POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz), POLYBENCH_ARRAY(_fict_));

    cudaSetDevice(GPU_DEVICE);
    fdtdCuda(tmax, nx, ny, POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz));

    if (verbose)
    {
        FILE *output_file = fopen("c_fdtd2d_output.txt", "w");
        if (output_file != NULL)
        {
            polybench_prevent_dce(print_array_to_file(output_file, nx, ny, POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz)));
            fclose(output_file);
        }
    }

    POLYBENCH_FREE_ARRAY(ex); POLYBENCH_FREE_ARRAY(ey); POLYBENCH_FREE_ARRAY(hz); POLYBENCH_FREE_ARRAY(_fict_);

    return 0;
}