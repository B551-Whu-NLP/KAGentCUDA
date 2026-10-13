#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void copy_array_d2d(double *src, double *dst, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int index = i * n + j;
            dst[index] = src[index];
        }
    }
}

int main(int argc, char *argv[]) {
    const int m = 1000;
    const int n = 1000;
    double* h_src = (double*)malloc(m * n * sizeof(double));
    double* h_dst = (double*)malloc(m * n * sizeof(double));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_src[i * n + j] = i * n + j;
        }
    }

    clock_t start = clock();

    for (int iter = 0; iter < 1000; iter++) {
        copy_array_d2d(h_src, h_dst, m, n);
    }

    clock_t end = clock();

    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_spent * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_74_output.txt");
        FILE *f = fopen(output_file, "w");
        if (f != NULL) {
            for (int i = 0; i < 100; ++i) {
                for (int j = 0; j < 100; ++j) {
                    fprintf(f, "%.2f ", h_src[i * n + j]);
                }
                fprintf(f, "\n");
            }
            fprintf(f, "\n");
            for (int i = 0; i < 100; ++i) {
                for (int j = 0; j < 100; ++j) {
                    fprintf(f, "%.2f ", h_dst[i * n + j]);
                }
                fprintf(f, "\n");
            }
            fclose(f);
        }
    }

    free(h_src);
    free(h_dst);

    return 0;
}