#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "mvt.h"

static
void init_array(int n,
    DATA_TYPE POLYBENCH_1D(x1,N,n),
    DATA_TYPE POLYBENCH_1D(x2,N,n),
    DATA_TYPE POLYBENCH_1D(y_1,N,n),
    DATA_TYPE POLYBENCH_1D(y_2,N,n),
    DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;
  for (i = 0; i < n; i++)
    {
      x1[i] = (DATA_TYPE) (i % n) / n;
      x2[i] = (DATA_TYPE) ((i + 1) % n) / n;
      y_1[i] = (DATA_TYPE) ((i + 3) % n) / n;
      y_2[i] = (DATA_TYPE) ((i + 4) % n) / n;
      for (j = 0; j < n; j++)
        A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }
}

static
void print_array_to_file(FILE* target, int n,
      DATA_TYPE POLYBENCH_1D(x1,N,n),
      DATA_TYPE POLYBENCH_1D(x2,N,n))
{
  int i;
  for (i = 0; i < n; i++) {
    fprintf (target, DATA_PRINTF_MODIFIER, x1[i]);
    if ((i + 1) % 20 == 0) fprintf (target, "\n");
  }
  for (i = 0; i < n; i++) {
    fprintf (target, DATA_PRINTF_MODIFIER, x2[i]);
    if ((i + 1) % 20 == 0) fprintf (target, "\n");
  }
}

static
void kernel_mvt(int n,
    DATA_TYPE POLYBENCH_1D(x1,N,n),
    DATA_TYPE POLYBENCH_1D(x2,N,n),
    DATA_TYPE POLYBENCH_1D(y_1,N,n),
    DATA_TYPE POLYBENCH_1D(y_2,N,n),
    DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;
#pragma scop
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x1[i] = x1[i] + A[i][j] * y_1[j];
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];
#pragma endscop
}

int main(int argc, char** argv)
{
  int verbose = 0;
  if (argc > 1 && strcmp(argv[1], "--verbose") == 0) verbose = 1;

  int n = N;

  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_2, DATA_TYPE, N, n);

  init_array (n,
        POLYBENCH_ARRAY(x1),
        POLYBENCH_ARRAY(x2),
        POLYBENCH_ARRAY(y_1),
        POLYBENCH_ARRAY(y_2),
        POLYBENCH_ARRAY(A));

  polybench_start_instruments;
  kernel_mvt (n,
        POLYBENCH_ARRAY(x1),
        POLYBENCH_ARRAY(x2),
        POLYBENCH_ARRAY(y_1),
        POLYBENCH_ARRAY(y_2),
        POLYBENCH_ARRAY(A));
  polybench_stop_instruments;
  polybench_print_instruments;

  if (verbose) {
    FILE *output_file = fopen("c_mvt_output.txt", "w");
    if (output_file != NULL) {
      polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2)));
      fclose(output_file);
    }
  }


  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x1);
  POLYBENCH_FREE_ARRAY(x2);
  POLYBENCH_FREE_ARRAY(y_1);
  POLYBENCH_FREE_ARRAY(y_2);

  return 0;
}