#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void matVecRowSub(const double* mat, const double* vec, double* buf, int m, int n) {
    for (int index = 0; index < m * n; index++) {
        int i = index / n;
        int j = index % n;
        buf[i * n + j] = mat[i * n + j] - vec[j];
    }
}

int main(int argc, char* argv[]) {
    int m = 5000;
    int n = 5000;
    double* h_mat = (double*)malloc(m * n * sizeof(double));
    double* h_vec = (double*)malloc(n * sizeof(double));
    double* h_buf = (double*)malloc(m * n * sizeof(double));

    for (int i = 0; i < m * n; ++i) {
        h_mat[i] = i * 0.1;
    }
    for (int i = 0; i < n; ++i) {
        h_vec[i] = i * 0.5;
    }

    clock_t start = clock();

    matVecRowSub(h_mat, h_vec, h_buf, m, n);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_278_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 100; ++i) {
                for (int j = 0; j < 100; ++j) {
                    fprintf(file, "%.2f ", h_buf[i * n + j]);
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    free(h_mat);
    free(h_vec);
    free(h_buf);

    return 0;
}