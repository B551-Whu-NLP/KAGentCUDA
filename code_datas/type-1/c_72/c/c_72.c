#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void matPerRowDivInplace(double *mat, const double *alphas, int m, int n) {
    for (int index = 0; index < m * n; index++) {
        int i = index / n;
        int j = index % n;
        mat[i * n + j] /= (alphas[i] + 10 * 3);
    }
}

int main(int argc, char *argv[]) {
    const int m = 10000;
    const int n = 10000;
    double* h_mat = (double*)malloc(m * n * sizeof(double));
    double* h_alphas = (double*)malloc(m * sizeof(double));
    
    for (int i = 0; i < m * n; ++i) {
        h_mat[i] = i + 1;
    }
    for (int i = 0; i < m; ++i) {
        h_alphas[i] = i + 1;
    }

    clock_t start = clock();

    matPerRowDivInplace(h_mat, h_alphas, m, n);

    clock_t end = clock();
    
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", timeElapsed);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_72_output.txt");
        FILE *fp = fopen(outputFileName, "w");
        if (fp != NULL) {
            for (int i = 0; i < 100; ++i) {
                for (int j = 0; j < 100; ++j) {
                    fprintf(fp, "%.6f ", h_mat[i * n + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_mat);
    free(h_alphas);

    return 0;
}