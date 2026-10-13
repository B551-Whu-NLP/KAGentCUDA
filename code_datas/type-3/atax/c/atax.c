#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "atax.h"

static
void init_array (int m, int n,
      DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
      DATA_TYPE POLYBENCH_1D(x,N,n))
{
  int i, j;
  DATA_TYPE fn;
  fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
      x[i] = 1 + (i / fn);
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % n) / (5*m);
}

static
void print_array_to_file(FILE* target, int n,
      DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i;
  for (i = 0; i < n; i++) {
    fprintf (target, DATA_PRINTF_MODIFIER, y[i]);
    if ((i + 1) % 20 == 0) fprintf (target, "\n");
  }
}

static
void kernel_atax(int m, int n,
      DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
      DATA_TYPE POLYBENCH_1D(x,N,n),
      DATA_TYPE POLYBENCH_1D(y,N,n),
      DATA_TYPE POLYBENCH_1D(tmp,M,m))
{
  int i, j;
#pragma scop
  for (i = 0; i < _PB_N; i++)
    y[i] = 0;
  for (i = 0; i < _PB_M; i++)
    {
      tmp[i] = SCALAR_VAL(0.0);
      for (j = 0; j < _PB_N; j++)
        tmp[i] = tmp[i] + A[i][j] * x[j];
      for (j = 0; j < _PB_N; j++)
        y[j] = y[j] + A[i][j] * tmp[i];
    }
#pragma endscop
}

int main(int argc, char** argv)
{
  int verbose = 0;
  if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
    verbose = 1;
  }

  int m = M;
  int n = N;

  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, N, m, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, M, m);

  init_array (m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

  polybench_start_instruments;
  kernel_atax (m, n,
          POLYBENCH_ARRAY(A),
          POLYBENCH_ARRAY(x),
          POLYBENCH_ARRAY(y),
          POLYBENCH_ARRAY(tmp));
  polybench_stop_instruments;
  polybench_print_instruments;

  if (verbose) {
    FILE *output_file = fopen("c_atax_output.txt", "w");
    if (output_file != NULL) {
      polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(y)));
      fclose(output_file);
    }
  }

  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(tmp);

  return 0;
}