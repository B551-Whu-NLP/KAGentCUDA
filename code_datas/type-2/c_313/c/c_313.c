#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void solveLower(const double* lower, const double* b, double* buf, int dim, int n) {
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < dim; i++) {
            double val = b[k * dim + i];
            for (int j = 0; j < i; j++) {
                val -= lower[i * dim + j] * buf[k * dim + j];
            }
            buf[k * dim + i] = val / lower[i * dim + i];
        }
    }
}

int main(int argc, char *argv[]) {
    int dim = 1000;
    int n = 1000; 

    double *lower = (double *)malloc(dim * dim * sizeof(double));
    double *b = (double *)malloc(n * dim * sizeof(double));
    double *buf = (double *)malloc(n * dim * sizeof(double));

    srand(42);
    for (int i = 0; i < dim * dim; i++) {
        lower[i] = rand() % 100 + 1.0; 
    }

    for (int i = 0; i < n * dim; i++) {
        b[i] = rand() % 100 + 1.0; 
    }

    clock_t start = clock();

    solveLower(lower, b, buf, dim, n);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "%s_output.txt", argv[0]);
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < dim; j++) {
                    fprintf(fp, "%.2f ", buf[i * dim + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(lower);
    free(b);
    free(buf);

    return 0;
}