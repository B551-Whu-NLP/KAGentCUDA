#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "gramschmidt.h"

static
void init_array(int m, int n,
    DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
    DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
    DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      A[i][j] = (((DATA_TYPE) ((i*j) % m) / m )*100) + 10;
      Q[i][j] = 0.0;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      R[i][j] = 0.0;
}

static
void print_array_to_file(FILE* target, int m, int n,
      DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
      DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf (target, DATA_PRINTF_MODIFIER, R[i][j]);
      if ((i * n + j + 1) % 20 == 0) fprintf (target, "\n");
    }
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      fprintf (target, DATA_PRINTF_MODIFIER, Q[i][j]);
      if ((i * n + j + 1) % 20 == 0) fprintf (target, "\n");
    }
}

static
void kernel_gramschmidt(int m, int n,
      DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
      DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
      DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j, k;
  DATA_TYPE nrm;
#pragma scop
  for (k = 0; k < _PB_N; k++)
    {
      nrm = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_M; i++)
        nrm += A[i][k] * A[i][k];
      R[k][k] = SQRT_FUN(nrm);
      for (i = 0; i < _PB_M; i++)
        Q[i][k] = A[i][k] / R[k][k];
      for (j = k + 1; j < _PB_N; j++)
        {
          R[k][j] = SCALAR_VAL(0.0);
          for (i = 0; i < _PB_M; i++)
            R[k][j] += Q[i][k] * A[i][j];
          for (i = 0; i < _PB_M; i++)
            A[i][j] = A[i][j] - Q[i][k] * R[k][j];
        }
    }
#pragma endscop
}

int main(int argc, char** argv)
{
  int verbose = 0;
  if (argc > 1 && strcmp(argv[1], "--verbose") == 0) verbose = 1;

  int m = M;
  int n = N;

  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(R,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(Q,DATA_TYPE,M,N,m,n);

  init_array (m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q));

  polybench_start_instruments;
  kernel_gramschmidt (m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q));
  polybench_stop_instruments;
  polybench_print_instruments;

  if (verbose) {
    FILE *output_file = fopen("c_gramschmidt_output.txt", "w");
    if (output_file != NULL) {
      polybench_prevent_dce(print_array_to_file(output_file, m, n, POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q)));
      fclose(output_file);
    }
  }


  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(R);
  POLYBENCH_FREE_ARRAY(Q);

  return 0;
}