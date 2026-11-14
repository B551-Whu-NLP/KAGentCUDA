#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void mxm_1d(double* a, const int m, double* b, const int n, double* c, const int p) {
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < p; k++) {
            double s = 0.0;
            for (int j = 0; j < n; j++) {
                s += a[j * m + i] * b[k * n + j];
            }
            c[k * m + i] = s;
        }
    }
}

int main(int argc, char *argv[]) {
    int m = 1024;
    int n = 512;
    int p = 256;

    double* h_a = (double*)malloc(m * n * sizeof(double));
    double* h_b = (double*)malloc(n * p * sizeof(double));
    double* h_c = (double*)malloc(m * p * sizeof(double));

    for (int i = 0; i < m * n; i++) {
        h_a[i] = i % 100;
    }
    for (int i = 0; i < n * p; i++) {
        h_b[i] = i % 50;
    }

    clock_t start = clock();

    mxm_1d(h_a, m, h_b, n, h_c, p);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[] = "c_109_output.txt";
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < p; i++) {
                for (int j = 0; j < m; j++) {
                    fprintf(fp, "%.2f ", h_c[i * m + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}