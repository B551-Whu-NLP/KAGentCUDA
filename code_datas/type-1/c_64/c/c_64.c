#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void matVecColAddInplace(double *mat, const double *vec, int m, int n) {
    for (int index = 0; index < m * n; index++) {
        int i = index / n;
        int j = index % n;
        mat[i * n + j] += vec[i];
    }
}

int main(int argc, char *argv[]) {
    const int m = 10000;
    const int n = 5000;
    double* h_mat = (double*)malloc(m * n * sizeof(double));
    double* h_vec = (double*)malloc(m * sizeof(double));

    for (int i = 0; i < m * n; ++i) {
        h_mat[i] = (double)(i);
    }

    for (int i = 0; i < m; ++i) {
        h_vec[i] = (double)(i * 2);
    }

    clock_t start = clock();

    matVecColAddInplace(h_mat, h_vec, m, n);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_64_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 100; ++i) {
                for (int j = 0; j < 50; ++j) {
                    fprintf(fp, "h_mat[%d][%d]: %f\n", i, j, h_mat[i * n + j]);
                }
            }
            fclose(fp);
        }
    }

    free(h_mat);
    free(h_vec);

    return 0;
}