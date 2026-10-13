#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "correlation.h"

static
void init_array (int m, int n, DATA_TYPE *float_n, DATA_TYPE POLYBENCH_2D(data,N,M,n,m))
{
  int i, j;
  *float_n = (DATA_TYPE)N;
  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = (DATA_TYPE)(i*j)/M + i;
}

static
void print_array_to_file(FILE* target, int m, DATA_TYPE POLYBENCH_2D(corr,M,M,m,m))
{
  int i, j;
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      fprintf (target, DATA_PRINTF_MODIFIER, corr[i][j]);
      if ((i * m + j + 1) % 20 == 0) fprintf (target, "\n");
    }
}

static
void kernel_correlation(int m, int n, DATA_TYPE float_n,
      DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
      DATA_TYPE POLYBENCH_2D(corr,M,M,m,m),
      DATA_TYPE POLYBENCH_1D(mean,M,m),
      DATA_TYPE POLYBENCH_1D(stddev,M,m))
{
  int i, j, k;
  DATA_TYPE eps = SCALAR_VAL(0.1);
#pragma scop
  for (j = 0; j < _PB_M; j++)
    {
      mean[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
        mean[j] += data[i][j];
      mean[j] /= float_n;
    }
   for (j = 0; j < _PB_M; j++)
    {
      stddev[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
        stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      stddev[j] /= float_n;
      stddev[j] = SQRT_FUN(stddev[j]);
      stddev[j] = stddev[j] <= eps ? SCALAR_VAL(1.0) : stddev[j];
    }
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      {
        data[i][j] -= mean[j];
        data[i][j] /= SQRT_FUN(float_n) * stddev[j];
      }
  for (i = 0; i < _PB_M-1; i++)
    {
      corr[i][i] = SCALAR_VAL(1.0);
      for (j = i+1; j < _PB_M; j++)
        {
          corr[i][j] = SCALAR_VAL(0.0);
          for (k = 0; k < _PB_N; k++)
            corr[i][j] += (data[k][i] * data[k][j]);
          corr[j][i] = corr[i][j];
        }
    }
  corr[_PB_M-1][_PB_M-1] = SCALAR_VAL(1.0);
#pragma endscop
}

int main(int argc, char** argv)
{
  int verbose = 0;
  if (argc > 1 && strcmp(argv[1], "--verbose") == 0) verbose = 1;

  int n = N;
  int m = M;

  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(corr,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE,M,m);

  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));

  polybench_start_instruments;
  kernel_correlation (m, n, float_n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(corr), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(stddev));
  polybench_stop_instruments;
  polybench_print_instruments;

  if (verbose) {
    FILE *output_file = fopen("c_correlation_output.txt", "w");
    if (output_file != NULL) {
      polybench_prevent_dce(print_array_to_file(output_file, m, POLYBENCH_ARRAY(corr)));
      fclose(output_file);
    }
  }


  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(corr);
  POLYBENCH_FREE_ARRAY(mean);
  POLYBENCH_FREE_ARRAY(stddev);

  return 0;
}