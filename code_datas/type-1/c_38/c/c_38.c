#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void mul(int N, float* X, int INCX, float* Y, int INCY) {
    for (int i = 0; i < N; ++i) {
        Y[i * INCY] *= X[i * INCX];
    }
}

int main(int argc, char* argv[]) {
    const int arraySize = 100000000; // 进一步增大数组大小
    float* h_X = (float*)malloc(arraySize * sizeof(float));
    float* h_Y = (float*)malloc(arraySize * sizeof(float));
    
    for (int i = 0; i < arraySize; ++i) {
        h_X[i] = (float)(i);
        h_Y[i] = (float)(2 * i);
    }

    clock_t start = clock();
    mul(arraySize, h_X, 1, h_Y, 1);
    clock_t end = clock();
    double timeTaken = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;

    printf("Time: %.2f(ms)\n", timeTaken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_38_output.txt");
        FILE* outputFile = fopen(outputFileName, "w");
        if (outputFile != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(outputFile, "%.2f ", h_Y[i]);
            }
            fclose(outputFile);
        }
    }

    free(h_X);
    free(h_Y);
    return 0;
}