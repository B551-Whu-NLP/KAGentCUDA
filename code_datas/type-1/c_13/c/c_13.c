#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void matDiagAddInplace(double* mat, double alpha, int dim) {
    for (int i = 0; i < dim; i++) {
        mat[i * dim + i] += alpha;
    }
}

int main(int argc, char* argv[]) {
    int matrixDim = 10000;
    double alpha = 10.0;
    double* h_mat = (double*)malloc(matrixDim * matrixDim * sizeof(double));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    srand(42);
    for (int i = 0; i < matrixDim * matrixDim; ++i) {
        h_mat[i] = (double)(rand() % 100);
    }
    
    clock_t start = clock();
    matDiagAddInplace(h_mat, alpha, matrixDim);
    clock_t end = clock();
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", timeElapsed);

    if (verbose) {
        FILE* file = fopen("c_13_output.txt", "w");
        if (file != NULL) {
            for (int i = 0; i < 100; i++) {
                for (int j = 0; j < 100; j++) {
                    fprintf(file, "%.2f ", h_mat[i * matrixDim + j]);
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    free(h_mat);
    return 0;
}