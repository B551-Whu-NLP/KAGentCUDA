#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void scal(int N, float ALPHA, float* X, int INCX) {
    int i;
    for (i = 0; i < N; ++i) {
        X[i * INCX] *= ALPHA;
    }
}

int main(int argc, char *argv[]) {
    int arraySize = 1000000;
    float ALPHA = 2.0;
    int INCX = 2;
    float* h_X = (float*)malloc(arraySize * INCX * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    srand(42);
    for (int i = 0; i < arraySize; ++i) {
        h_X[i] = (float)(rand() % 100);
    }

    clock_t start = clock();
    scal(arraySize, ALPHA, h_X, INCX);
    clock_t end = clock();

    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    if (verbose) {
        char outputFileName[256];
        sprintf(outputFileName, "%s_output.txt", argv[0]);
        FILE *fp = fopen(outputFileName, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", h_X[i]);
            }
            fclose(fp);
        }
    }

    free(h_X);

    printf("Time: %.2f(ms)\n", timeElapsed);
    return 0;
}