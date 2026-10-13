#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "bicg.h"

static
void init_array (int m, int n,
      DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
      DATA_TYPE POLYBENCH_1D(r,N,n),
      DATA_TYPE POLYBENCH_1D(p,M,m))
{
  int i, j;
  for (i = 0; i < m; i++)
    p[i] = (DATA_TYPE)(i % m) / m;
  for (i = 0; i < n; i++) {
    r[i] = (DATA_TYPE)(i % n) / n;
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % n)/n;
  }
}

static
void print_array_to_file(FILE* target, int m, int n,
      DATA_TYPE POLYBENCH_1D(s,M,m),
      DATA_TYPE POLYBENCH_1D(q,N,n))
{
  int i;
  for (i = 0; i < m; i++) {
    fprintf (target, DATA_PRINTF_MODIFIER, s[i]);
    if ((i + 1) % 20 == 0) fprintf (target, "\n");
  }
  for (i = 0; i < n; i++) {
    fprintf (target, DATA_PRINTF_MODIFIER, q[i]);
    if ((i + 1) % 20 == 0) fprintf (target, "\n");
  }
}

static
void kernel_bicg(int m, int n,
      DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
      DATA_TYPE POLYBENCH_1D(s,M,m),
      DATA_TYPE POLYBENCH_1D(q,N,n),
      DATA_TYPE POLYBENCH_1D(p,M,m),
      DATA_TYPE POLYBENCH_1D(r,N,n))
{
  int i, j;
#pragma scop
  for (i = 0; i < _PB_M; i++)
    s[i] = 0;
  for (i = 0; i < _PB_N; i++)
    {
      q[i] = SCALAR_VAL(0.0);
      for (j = 0; j < _PB_M; j++)
  {
    s[j] = s[j] + r[i] * A[i][j];
    q[i] = q[i] + A[i][j] * p[j];
  }
    }
#pragma endscop
}

int main(int argc, char** argv)
{
  int verbose = 0;
  if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
    verbose = 1;
  }

  int n = N;
  int m = M;

  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, M, n, m);
  POLYBENCH_1D_ARRAY_DECL(s, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(q, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);

  init_array (m, n,
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(r),
        POLYBENCH_ARRAY(p));

  polybench_start_instruments;
  kernel_bicg (m, n,
          POLYBENCH_ARRAY(A),
          POLYBENCH_ARRAY(s),
          POLYBENCH_ARRAY(q),
          POLYBENCH_ARRAY(p),
          POLYBENCH_ARRAY(r));
  polybench_stop_instruments;
  polybench_print_instruments;

  if (verbose) {
    FILE *output_file = fopen("c_bicg_output.txt", "w");
    if (output_file != NULL) {
      polybench_prevent_dce(print_array_to_file(output_file, m, n, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(q)));
      fclose(output_file);
    }
  }

  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(s);
  POLYBENCH_FREE_ARRAY(q);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(r);

  return 0;
}