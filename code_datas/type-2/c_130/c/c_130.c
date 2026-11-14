#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void colLog2SumExp2(const double* mat, double* buf, int m, int n) {
    for (int j = 0; j < n; j++) {
        double maximum = mat[j];

        for (int i = 1; i < m; i++) {
            if (mat[i * n + j] > maximum) {
                maximum = mat[i * n + j];
            }
        }

        double res = 0.0;

        for (int i = 0; i < m; i++) {
            res += exp(mat[i * n + j] - maximum);
        }

        buf[j] = log2(res) + maximum;
    }
}

int main(int argc, char** argv) {
    int m = 10000;
    int n = 5000;

    double* h_mat = (double*)malloc(m * n * sizeof(double));
    double* h_buf = (double*)malloc(n * sizeof(double));

    srand(42);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            h_mat[i * n + j] = rand() / (double)RAND_MAX * 10.0;
        }
    }

    clock_t start = clock();
    colLog2SumExp2(h_mat, h_buf, m, n);
    clock_t end = clock();

    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", timeElapsed);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_130_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int j = 0; j < 1000; j++) {
                fprintf(file, "%f\n", h_buf[j]);
            }
            fclose(file);
        }
    }

    free(h_mat);
    free(h_buf);

    return 0;
}