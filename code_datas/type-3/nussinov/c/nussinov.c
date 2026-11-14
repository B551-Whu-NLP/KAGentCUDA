#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "nussinov.h"

typedef char base;
#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0)
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

static
void init_array (int n,
                 base POLYBENCH_1D(seq,N,n),
                 DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
  int i, j;
  for (i=0; i <n; i++) {
     seq[i] = (base)((i+1)%4);
  }
  for (i=0; i <n; i++)
     for (j=0; j <n; j++)
        table[i][j] = 0;
}

static
void print_array_to_file(FILE* target, int n,
       DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
  int i, j;
  int t = 0;
  for (i = 0; i < n; i++) {
    for (j = i; j < n; j++) {
      fprintf (target, DATA_PRINTF_MODIFIER, table[i][j]);
      if ((t + 1) % 20 == 0) fprintf (target, "\n");
      t++;
    }
  }
}

static
void kernel_nussinov(int n, base POLYBENCH_1D(seq,N,n),
            DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
  int i, j, k;
#pragma scop
 for (i = _PB_N-1; i >= 0; i--) {
  for (j=i+1; j<_PB_N; j++) {
   if (j-1>=0)
      table[i][j] = max_score(table[i][j], table[i][j-1]);
   if (i+1<_PB_N)
      table[i][j] = max_score(table[i][j], table[i+1][j]);
   if (j-1>=0 && i+1<_PB_N) {
     if (i<j-1)
        table[i][j] = max_score(table[i][j], table[i+1][j-1]+match(seq[i], seq[j]));
     else
        table[i][j] = max_score(table[i][j], table[i+1][j-1]);
   }
   for (k=i+1; k<j; k++) {
      table[i][j] = max_score(table[i][j], table[i][k] + table[k+1][j]);
   }
  }
 }
#pragma endscop
}

int main(int argc, char** argv)
{
  int verbose = 0;
  if (argc > 1 && strcmp(argv[1], "--verbose") == 0) verbose = 1;

  int n = N;

  POLYBENCH_1D_ARRAY_DECL(seq, base, N, n);
  POLYBENCH_2D_ARRAY_DECL(table, DATA_TYPE, N, N, n, n);

  init_array (n, POLYBENCH_ARRAY(seq), POLYBENCH_ARRAY(table));

  polybench_start_instruments;
  kernel_nussinov (n, POLYBENCH_ARRAY(seq), POLYBENCH_ARRAY(table));
  polybench_stop_instruments;
  polybench_print_instruments;

  if (verbose) {
    FILE *output_file = fopen("c_nussinov_output.txt", "w");
    if (output_file != NULL) {
      polybench_prevent_dce(print_array_to_file(output_file, n, POLYBENCH_ARRAY(table)));
      fclose(output_file);
    }
  }


  POLYBENCH_FREE_ARRAY(seq);
  POLYBENCH_FREE_ARRAY(table);

  return 0;
}