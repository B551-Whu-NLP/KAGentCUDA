#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void matrixMultiply(float* A, float* B, float* C, int numARows, int numAColumns, int numBRows, int numBColumns) {
    int numCRows = numARows;
    int numCColumns = numBColumns;

    for (int row = 0; row < numCRows; row++) {
        for (int col = 0; col < numCColumns; col++) {
            float sum = 0;

            for (int k = 0; k < numBRows; k++) {
                sum += A[row * numAColumns + k] * B[k * numBColumns + col];
            }

            C[row * numCColumns + col] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    int numARows = 1500; 
    int numAColumns = 1200;
    int numBRows = 1200;
    int numBColumns = 900; 

    float *A = (float *)malloc(numARows * numAColumns * sizeof(float));
    float *B = (float *)malloc(numBRows * numBColumns * sizeof(float));
    float *C = (float *)malloc(numARows * numBColumns * sizeof(float));

    srand(42);
    for (int i = 0; i < numARows * numAColumns; i++) {
        A[i] = rand() % 100 + 1.0; 
    }

    srand(123);
    for (int i = 0; i < numBRows * numBColumns; i++) {
        B[i] = rand() % 100 + 1.0; 
    }

    clock_t start = clock();
    matrixMultiply(A, B, C, numARows, numAColumns, numBRows, numBColumns);
    clock_t end = clock();

    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", timeElapsed);

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_316_output.txt");
        FILE *fp = fopen(outputFileName, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                for (int j = 0; j < numBColumns; j++) {
                    fprintf(fp, "%.2f ", C[i * numBColumns + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(A);
    free(B);
    free(C);

    return 0;
}