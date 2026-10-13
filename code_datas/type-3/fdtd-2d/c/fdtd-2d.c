#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "fdtd-2d.h"

static
void init_array (int tmax, int nx, int ny,
      DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
      DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
      DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
      DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  int i, j;
  for (i = 0; i < tmax; i++)
    _fict_[i] = (DATA_TYPE) i;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      {
        ex[i][j] = ((DATA_TYPE) i*(j+1)) / nx;
        ey[i][j] = ((DATA_TYPE) i*(j+2)) / ny;
        hz[i][j] = ((DATA_TYPE) i*(j+3)) / nx;
      }
}

static
void print_array_to_file(FILE* target, int nx, int ny,
      DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
      DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
      DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
  int i, j;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      fprintf(target, DATA_PRINTF_MODIFIER, ex[i][j]);
      if ((i * nx + j + 1) % 20 == 0) fprintf(target, "\n");
    }
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      fprintf(target, DATA_PRINTF_MODIFIER, ey[i][j]);
      if ((i * nx + j + 1) % 20 == 0) fprintf(target, "\n");
    }
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      fprintf(target, DATA_PRINTF_MODIFIER, hz[i][j]);
      if ((i * nx + j + 1) % 20 == 0) fprintf(target, "\n");
    }
}

static
void kernel_fdtd_2d(int tmax, int nx, int ny,
        DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
        DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
        DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
        DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  int t, i, j;
#pragma scop
  for(t = 0; t < _PB_TMAX; t++)
    {
      for (j = 0; j < _PB_NY; j++)
        ey[0][j] = _fict_[t];
      for (i = 1; i < _PB_NX; i++)
        for (j = 0; j < _PB_NY; j++)
          ey[i][j] = ey[i][j] - SCALAR_VAL(0.5)*(hz[i][j]-hz[i-1][j]);
      for (i = 0; i < _PB_NX; i++)
        for (j = 1; j < _PB_NY; j++)
          ex[i][j] = ex[i][j] - SCALAR_VAL(0.5)*(hz[i][j]-hz[i][j-1]);
      for (i = 0; i < _PB_NX - 1; i++)
        for (j = 0; j < _PB_NY - 1; j++)
          hz[i][j] = hz[i][j] - SCALAR_VAL(0.7)* (ex[i][j+1] - ex[i][j] +
                       ey[i+1][j] - ey[i][j]);
    }
#pragma endscop
}

int main(int argc, char** argv)
{
  int verbose = 0;
  if (argc > 1 && strcmp(argv[1], "--verbose") == 0) verbose = 1;

  int tmax = TMAX;
  int nx = NX;
  int ny = NY;

  POLYBENCH_2D_ARRAY_DECL(ex,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(ey,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(hz,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_1D_ARRAY_DECL(_fict_,DATA_TYPE,TMAX,tmax);

  init_array (tmax, nx, ny, POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz), POLYBENCH_ARRAY(_fict_));

  polybench_start_instruments;
  kernel_fdtd_2d (tmax, nx, ny, POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz), POLYBENCH_ARRAY(_fict_));
  polybench_stop_instruments;
  polybench_print_instruments;

  if (verbose) {
    FILE *output_file = fopen("c_fdtd2d_output.txt", "w");
    if (output_file != NULL) {
      polybench_prevent_dce(print_array_to_file(output_file, nx, ny, POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz)));
      fclose(output_file);
    }
  }


  POLYBENCH_FREE_ARRAY(ex);
  POLYBENCH_FREE_ARRAY(ey);
  POLYBENCH_FREE_ARRAY(hz);
  POLYBENCH_FREE_ARRAY(_fict_);

  return 0;
}