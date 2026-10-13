#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

double matVecRowSubInplace(double *mat, const double *vec, int m, int n) {
    clock_t start = clock();

    for (int index = 0; index < m * n; index++) {
        int i = index / n;
        int j = index % n;
        mat[i * n + j] -= vec[j];
    }

    clock_t end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
}

int main(int argc, char *argv[]) {
    const int m = 1000;
    const int n = 500;
    double* h_mat = (double*)malloc(m * n * sizeof(double));
    double* h_vec = (double*)malloc(n * sizeof(double));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < m * n; ++i) {
        h_mat[i] = (double)(i % 100);
    }

    for (int i = 0; i < n; ++i) {
        h_vec[i] = (double)(i % 50);
    }

    double time_taken = matVecRowSubInplace(h_mat, h_vec, m, n);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_63_output.txt");
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
    printf("Time: %.2f(ms)\n", time_taken);

    return 0;
}