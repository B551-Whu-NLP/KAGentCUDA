#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "symm.h"

static
void init_array(int m, int n,
    DATA_TYPE *alpha,
    DATA_TYPE *beta,
    DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
    DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
    DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;
  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
      B[i][j] = (DATA_TYPE) ((n+i-j) % 100) / m;
    }
  for (i = 0; i < m; i++) {
    for (j = 0; j <=i; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
    for (j = i+1; j < m; j++)
      A[i][j] = -999;
  }
}

static
void print_array_to_file(FILE* target, int m, int n,
     DATA_TYPE POLYBENCH_2D(C,M,N,m,n))
{
  int i, j;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      fprintf (target, DATA_PRINTF_MODIFIER, C[i][j]);
      if ((i * m + j + 1) % 20 == 0) fprintf (target, "\n");
    }
}

static
void kernel_symm(int m, int n,
     DATA_TYPE alpha,
     DATA_TYPE beta,
     DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
     DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
     DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j, k;
  DATA_TYPE temp2;
#pragma scop
   for (i = 0; i < _PB_M; i++)
      for (j = 0; j < _PB_N; j++ )
      {
        temp2 = 0;
        for (k = 0; k < i; k++) {
           C[k][j] += alpha*B[i][j] * A[i][k];
           temp2 += B[k][j] * A[i][k];
        }
        C[i][j] = beta * C[i][j] + alpha*B[i][j] * A[i][i] + alpha * temp2;
     }
#pragma endscop
}

int main(int argc, char** argv)
{
  int verbose = 0;
  if (argc > 1 && strcmp(argv[1], "--verbose") == 0) verbose = 1;

  int m = M;
  int n = N;

  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,M,m,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,M,N,m,n);

  init_array (m, n, &alpha, &beta,
        POLYBENCH_ARRAY(C),
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(B));

  polybench_start_instruments;
  kernel_symm (m, n, alpha, beta, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));
  polybench_stop_instruments;
  polybench_print_instruments;

  if (verbose) {
    FILE *output_file = fopen("c_symm_output.txt", "w");
    if (output_file != NULL) {
      polybench_prevent_dce(print_array_to_file(output_file, m, n, POLYBENCH_ARRAY(C)));
      fclose(output_file);
    }
  }


  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}