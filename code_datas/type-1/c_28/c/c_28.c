#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void dsubtract_matrix(double* a, double* b, double* c, int N) {
    for (int idx = 0; idx < N; idx++) {
        c[idx] = a[idx] - b[idx];
    }
}

int main(int argc, char** argv) {
    int arraySize = 1000000; // 增大数组大小
    double* h_a = (double*)malloc(arraySize * sizeof(double));
    double* h_b = (double*)malloc(arraySize * sizeof(double));
    double* h_c = (double*)malloc(arraySize * sizeof(double));

    for (int i = 0; i < arraySize; ++i) {
        h_a[i] = (double)(i);
        h_b[i] = (double)(2 * i);
    }

    clock_t start = clock();

    dsubtract_matrix(h_a, h_b, h_c, arraySize);

    clock_t end = clock();

    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", timeElapsed * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_28_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.2f ", h_c[i]);
            }
            fclose(file);
        }
    }

    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}