#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "covariance.h"

static
void init_array (int m, int n, DATA_TYPE *float_n, DATA_TYPE POLYBENCH_2D(data,N,M,n,m))
{
  int i, j;
  *float_n = (DATA_TYPE)n;
  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = ((DATA_TYPE) i*j) / M;
}

static
void print_array_to_file(FILE* target, int m, DATA_TYPE POLYBENCH_2D(cov,M,M,m,m))
{
  int i, j;
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      fprintf (target, DATA_PRINTF_MODIFIER, cov[i][j]);
      if ((i * m + j + 1) % 20 == 0) fprintf (target, "\n");
    }
}

static
void kernel_covariance(int m, int n, DATA_TYPE float_n,
           DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
           DATA_TYPE POLYBENCH_2D(cov,M,M,m,m),
           DATA_TYPE POLYBENCH_1D(mean,M,m))
{
  int i, j, k;
#pragma scop
  for (j = 0; j < _PB_M; j++)
    {
      mean[j] = SCALAR_VAL(0.0);
      for (i = 0; i < _PB_N; i++)
        mean[j] += data[i][j];
      mean[j] /= float_n;
    }
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_M; j++)
      data[i][j] -= mean[j];
  for (i = 0; i < _PB_M; i++)
    for (j = i; j < _PB_M; j++)
      {
        cov[i][j] = SCALAR_VAL(0.0);
        for (k = 0; k < _PB_N; k++)
          cov[i][j] += data[k][i] * data[k][j];
        cov[i][j] /= (float_n - SCALAR_VAL(1.0));
        cov[j][i] = cov[i][j];
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

  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(cov,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);

  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));

  polybench_start_instruments;
  kernel_covariance (m, n, float_n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(cov), POLYBENCH_ARRAY(mean));
  polybench_stop_instruments;
  polybench_print_instruments;

  if (verbose) {
    FILE *output_file = fopen("c_covariance_output.txt", "w");
    if (output_file != NULL) {
      polybench_prevent_dce(print_array_to_file(output_file, m, POLYBENCH_ARRAY(cov)));
      fclose(output_file);
    }
  }


  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(cov);
  POLYBENCH_FREE_ARRAY(mean);

  return 0;
}