#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "jacobi-1d.h"

static
void init_array (int n,
      DATA_TYPE POLYBENCH_1D(A,N,n),
      DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int i;
  for (i = 0; i < n; i++)
      {
  A[i] = ((DATA_TYPE) i+ 2) / n;
  B[i] = ((DATA_TYPE) i+ 3) / n;
      }
}

static
void print_array_to_file(FILE* target, int n,
      DATA_TYPE POLYBENCH_1D(A,N,n))
{
  int i;
  for (i = 0; i < n; i++)
    {
      fprintf(target, DATA_PRINTF_MODIFIER, A[i]);
      if ((i + 1) % 20 == 0) fprintf(target, "\n");
    }
}

static
void kernel_jacobi_1d(int tsteps,
          int n,
          DATA_TYPE POLYBENCH_1D(A,N,n),
          DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int t, i;
#pragma scop
  for (t = 0; t < _PB_TSTEPS; t++)
    {
      for (i = 1; i < _PB_N - 1; i++)
  B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
      for (i = 1; i < _PB_N - 1; i++)
  A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
    }
#pragma endscop
}

int main(int argc, char** argv)
{
  int verbose = 0;
  if (argc > 1 && strcmp(argv[1], "--verbose") == 0) verbose = 1;

  int n = N;
  int tsteps = TSTEPS;

  POLYBENCH_1D_ARRAY_DECL(A, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B, DATA_TYPE, N, n);

  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  polybench_start_instruments;
  kernel_jacobi_1d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));
  polybench_stop_instruments;
  polybench_print_instruments;

  if (verbose) {
    FILE *output_file = fopen("c_jacobi1d_output.txt", "w");
    if (output_file != NULL) {
      polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(A)));
      fclose(output_file);
    }
  }


  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}