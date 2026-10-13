#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void doubleArrayScalarDivide(double* d_in, int* d_out, int length, double scalar) {
    for (int idx = 0; idx < length; idx++) {
        d_out[idx] = d_in[idx] / scalar;
    }
}

int main(int argc, char* argv[]) {
    const int arraySize = 10000000;
    double* h_d_in = (double*)malloc(arraySize * sizeof(double));
    int* h_d_out = (int*)malloc(arraySize * sizeof(int));
    const double scalar = 2.0;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < arraySize; ++i) {
        h_d_in[i] = (double)(i);
    }

    clock_t start = clock();

    doubleArrayScalarDivide(h_d_in, h_d_out, arraySize, scalar);

    clock_t end = clock();

    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", timeElapsed);

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_33_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%d ", h_d_out[i]);
            }
            fclose(file);
        }
    }

    free(h_d_in);
    free(h_d_out);
    return 0;
}