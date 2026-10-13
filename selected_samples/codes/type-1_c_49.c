#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void copy(int N, float* X, int INCX, float* Y, int INCY, double* time_spent) {
    clock_t start = clock();

    for (int i = 0; i < N; ++i) {
        Y[i * INCY] = X[i * INCX];
    }

    clock_t end = clock();
    *time_spent = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
}

int main(int argc, char* argv[]) {
    const int arraySize = 10000000;
    float* h_X = (float*)malloc(arraySize * sizeof(float));
    float* h_Y = (float*)malloc(arraySize * sizeof(float));
    int verbose = 0;
    double time_spent;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < arraySize; ++i) {
        h_X[i] = (float)(i);
    }

    copy(arraySize, h_X, 1, h_Y, 1, &time_spent);

    if (verbose) {
        FILE* file = fopen("c_49_output.txt", "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.2f ", h_Y[i]);
            }
            fclose(file);
        }
    }

    free(h_X);
    free(h_Y);
    printf("Time: %.2f(ms)\n", time_spent);
    return 0;
}