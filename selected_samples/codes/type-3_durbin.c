#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "durbin.h"

static
void init_array (int n,
      DATA_TYPE POLYBENCH_1D(r,N,n))
{
  int i;
  for (i = 0; i < n; i++)
    {
      r[i] = (n+1-i);
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
void kernel_durbin(int n,
       DATA_TYPE POLYBENCH_1D(r,N,n),
       DATA_TYPE POLYBENCH_1D(y,N,n))
{
 DATA_TYPE z[N];
 DATA_TYPE alpha;
 DATA_TYPE beta;
 DATA_TYPE sum;
 int i,k;

#pragma scop
 y[0] = -r[0];
 beta = SCALAR_VAL(1.0);
 alpha = -r[0];

 for (k = 1; k < _PB_N; k++) {
   beta = (1-alpha*alpha)*beta;
   sum = SCALAR_VAL(0.0);
   for (i=0; i<k; i++) {
      sum += r[k-i-1]*y[i];
   }
   alpha = - (r[k] + sum)/beta;

   for (i=0; i<k; i++) {
      z[i] = y[i] + alpha*y[k-i-1];
   }
   for (i=0; i<k; i++) {
     y[i] = z[i];
   }
   y[k] = alpha;
 }
#pragma endscop
}

int main(int argc, char** argv)
{
  int verbose = 0;
  if (argc > 1 && strcmp(argv[1], "--verbose") == 0) verbose = 1;

  int n = N;

  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);

  init_array (n, POLYBENCH_ARRAY(r));

  polybench_start_instruments;
  kernel_durbin (n, POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(y));
  polybench_stop_instruments;
  polybench_print_instruments;

  if (verbose) {
    FILE *output_file = fopen("c_durbin_output.txt", "w");
    if (output_file != NULL) {
      polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(y)));
      fclose(output_file);
    }
  }



  POLYBENCH_FREE_ARRAY(r);
  POLYBENCH_FREE_ARRAY(y);

  return 0;
}