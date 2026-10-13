#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "gesummv.h"

static
void init_array(int n,
    DATA_TYPE *alpha,
    DATA_TYPE *beta,
    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
    DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
    DATA_TYPE POLYBENCH_1D(x,N,n))
{
  int i, j;
  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    {
      x[i] = (DATA_TYPE)( i % n) / n;
      for (j = 0; j < n; j++) {
        A[i][j] = (DATA_TYPE) ((i*j+1) % n) / n;
        B[i][j] = (DATA_TYPE) ((i*j+2) % n) / n;
      }
    }
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
void kernel_gesummv(int n,
        DATA_TYPE alpha,
        DATA_TYPE beta,
        DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
        DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
        DATA_TYPE POLYBENCH_1D(tmp,N,n),
        DATA_TYPE POLYBENCH_1D(x,N,n),
        DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j;
#pragma scop
  for (i = 0; i < _PB_N; i++)
    {
      tmp[i] = SCALAR_VAL(0.0);
      y[i] = SCALAR_VAL(0.0);
      for (j = 0; j < _PB_N; j++)
        {
          tmp[i] = A[i][j] * x[j] + tmp[i];
          y[i] = B[i][j] * x[j] + y[i];
        }
      y[i] = alpha * tmp[i] + beta * y[i];
    }
#pragma endscop
}

int main(int argc, char** argv)
{
  int verbose = 0;
  if (argc > 1 && strcmp(argv[1], "--verbose") == 0) verbose = 1;

  int n = N;

  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);

  init_array (n, &alpha, &beta,
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(B),
        POLYBENCH_ARRAY(x));

  polybench_start_instruments;
  kernel_gesummv (n, alpha, beta,
      POLYBENCH_ARRAY(A),
      POLYBENCH_ARRAY(B),
      POLYBENCH_ARRAY(tmp),
      POLYBENCH_ARRAY(x),
      POLYBENCH_ARRAY(y));
  polybench_stop_instruments;
  polybench_print_instruments;

  if (verbose) {
    FILE *output_file = fopen("c_gesummv_output.txt", "w");
    if (output_file != NULL) {
      polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(y)));
      fclose(output_file);
    }
  }



  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(tmp);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}