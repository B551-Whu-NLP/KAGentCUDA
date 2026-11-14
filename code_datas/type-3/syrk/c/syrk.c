#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "syrk.h"

static
void init_array(int n, int m,
    DATA_TYPE *alpha,
    DATA_TYPE *beta,
    DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
    DATA_TYPE POLYBENCH_2D(A,N,M,n,m))
{
  int i, j;
  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1)%n) / n;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      C[i][j] = (DATA_TYPE) ((i*j+2)%m) / m;
}

static
void print_array_to_file(FILE* target, int n,
     DATA_TYPE POLYBENCH_2D(C,N,N,n,n))
{
  int i, j;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf (target, DATA_PRINTF_MODIFIER, C[i][j]);
      if ((i * n + j + 1) % 20 == 0) fprintf (target, "\n");
    }
}

static
void kernel_syrk(int n, int m,
     DATA_TYPE alpha,
     DATA_TYPE beta,
     DATA_TYPE POLYBENCH_2D(C,N,N,n,n),
     DATA_TYPE POLYBENCH_2D(A,N,M,n,m))
{
  int i, j, k;
#pragma scop
  for (i = 0; i < _PB_N; i++) {
    for (j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (k = 0; k < _PB_M; k++) {
      for (j = 0; j <= i; j++)
        C[i][j] += alpha * A[i][k] * A[j][k];
    }
  }
#pragma endscop
}

int main(int argc, char** argv)
{
  int verbose = 0;
  if (argc > 1 && strcmp(argv[1], "--verbose") == 0) verbose = 1;

  int n = N;
  int m = M;

  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,M,n,m);

  init_array (n, m, &alpha, &beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));

  polybench_start_instruments;
  kernel_syrk (n, m, alpha, beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A));
  polybench_stop_instruments;
  polybench_print_instruments;

  if (verbose) {
    FILE *output_file = fopen("c_syrk_output.txt", "w");
    if (output_file != NULL) {
      polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(C)));
      fclose(output_file);
    }
  }


  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}