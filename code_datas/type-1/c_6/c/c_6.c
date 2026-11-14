#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void allAddInplace(double* arr, double alpha, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] += alpha;
    }
}

int main(int argc, char* argv[]) {
    int arraySize = 10000000; // 增大数组大小
    double alpha = 5.0;
    double* h_arr = (double*)malloc(arraySize * sizeof(double));

    for (int i = 0; i < arraySize; ++i) {
        h_arr[i] = (double)i;
    }

    clock_t start = clock();

    // 多次调用以增加总执行时间
    for (int iter = 0; iter < 100; iter++) {
        allAddInplace(h_arr, alpha, arraySize);
    }

    clock_t end = clock();

    double timeTaken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_6_output.txt");
        FILE* outputFile = fopen(outputFileName, "w");
        if (outputFile != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(outputFile, "%.2f ", h_arr[i]);
            }
            fclose(outputFile);
        }
    }

    free(h_arr);
    printf("Time: %.2f(ms)\n", timeTaken);
    return 0;
}