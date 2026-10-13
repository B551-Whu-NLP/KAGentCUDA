#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include "trmm.h"

static void init_array(int m, int n, DATA_TYPE *alpha,
                DATA_TYPE POLYBENCH_2D(A, M, M, m, m),
                DATA_TYPE POLYBENCH_2D(B, M, N, m, n))
{
    int i, j;
    *alpha = 1.5;
    for (i = 0; i < m; i++) {
        for (j = 0; j < i; j++) A[i][j] = (DATA_TYPE)((i + j) % m) / m;
        A[i][i] = 1.0;
        for (j = 0; j < n; j++) B[i][j] = (DATA_TYPE)((n + (i - j)) % n) / n;
    }
}

static void print_array_to_file(FILE* target, int m, int n, DATA_TYPE POLYBENCH_2D(B, M, N, m, n))
{
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++) {
            fprintf(target, DATA_PRINTF_MODIFIER, B[i][j]);
            if ((i * m + j + 1) % 20 == 0) fprintf(target, "\n");
        }
}

/**
 * CPU Implementation: B_out := alpha * (A^T * B_in)
 * A is an upper triangular matrix after transpose (original A is lower triangular)
 */
void trmm_cpu(int m, int n, DATA_TYPE alpha, DATA_TYPE *A, DATA_TYPE *B_in, DATA_TYPE *B_out)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            DATA_TYPE dot = SCALAR_VAL(0.0);
            // A is lower triangular, A^T is upper triangular. 
            // A[i][i] is unit (1.0)
            for (int k = i + 1; k < m; k++) {
                dot += A[k * m + i] * B_in[k * n + j];
            }
            B_out[i * n + j] = alpha * (B_in[i * n + j] + dot);
        }
    }
}

int main(int argc, char** argv)
{
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) verbose = 1;

    int m = M;
    int n = N;

    DATA_TYPE alpha;
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, M, m, m);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, M, N, m, n);

    init_array(m, n, &alpha, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

    DATA_TYPE *B_out = (DATA_TYPE *)malloc(M * N * sizeof(DATA_TYPE));

    polybench_start_instruments;
    trmm_cpu(m, n, alpha, &POLYBENCH_ARRAY(A)[0][0], &POLYBENCH_ARRAY(B)[0][0], B_out);
    polybench_stop_instruments;
    polybench_print_instruments;

    // Copy result back to B
    memcpy(&POLYBENCH_ARRAY(B)[0][0], B_out, M * N * sizeof(DATA_TYPE));

    if (verbose) {
        FILE *output_file = fopen("c_trmm_output.txt", "w");
        if (output_file != NULL) {
            polybench_prevent_dce(print_array_to_file(output_file, m, n, POLYBENCH_ARRAY(B)));
            fclose(output_file);
        }
    }

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    free(B_out);

    return 0;
}
