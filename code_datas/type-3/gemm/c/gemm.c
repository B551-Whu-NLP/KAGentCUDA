#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "gemm.h"

static
void init_array(int ni, int nj, int nk,
    DATA_TYPE *alpha,
    DATA_TYPE *beta,
    DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
    DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
    DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j;
  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % nk) / nk;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+2) % nj) / nj;
}

static
void print_array_to_file(FILE* target, int ni, int nj,
     DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
      fprintf (target, DATA_PRINTF_MODIFIER, C[i][j]);
      if ((i * ni + j + 1) % 20 == 0) fprintf (target, "\n");
    }
}

static
void kernel_gemm(int ni, int nj, int nk,
     DATA_TYPE alpha,
     DATA_TYPE beta,
     DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
     DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
     DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j, k;
#pragma scop
  for (i = 0; i < _PB_NI; i++) {
    for (j = 0; j < _PB_NJ; j++)
      C[i][j] *= beta;
    for (k = 0; k < _PB_NK; k++) {
       for (j = 0; j < _PB_NJ; j++)
        C[i][j] += alpha * A[i][k] * B[k][j];
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

  int ni = NI;
  int nj = NJ;
  int nk = NK;

  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NJ,ni,nj);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);

  init_array (ni, nj, nk, &alpha, &beta,
        POLYBENCH_ARRAY(C),
        POLYBENCH_ARRAY(A),
        POLYBENCH_ARRAY(B));

  polybench_start_instruments;
  kernel_gemm (ni, nj, nk,
          alpha, beta,
          POLYBENCH_ARRAY(C),
          POLYBENCH_ARRAY(A),
          POLYBENCH_ARRAY(B));
  polybench_stop_instruments;
  polybench_print_instruments;

  if (verbose) {
    FILE *output_file = fopen("c_gemm_output.txt", "w");
    if (output_file != NULL) {
      polybench_prevent_dce(print_array_to_file(output_file, ni, nj, POLYBENCH_ARRAY(C)));
      fclose(output_file);
    }
  }


  POLYBENCH_FREE_ARRAY(C);
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}