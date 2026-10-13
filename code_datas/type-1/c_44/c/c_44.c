#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void add(float* c, float* a, float* b, int n) {
    for (int k = 0; k < n; k++) {
        c[k] = a[k] + b[k];
    }
}

int main(int argc, char* argv[]) {
    const int arraySize = 10000000; // 增大数组大小
    float* h_a = (float*)malloc(arraySize * sizeof(float));
    float* h_b = (float*)malloc(arraySize * sizeof(float));
    float* h_c = (float*)malloc(arraySize * sizeof(float));

    for (int i = 0; i < arraySize; ++i) {
        h_a[i] = (float)(i);
        h_b[i] = (float)(2 * i);
    }

    clock_t start = clock();
    add(h_c, h_a, h_b, arraySize);
    clock_t end = clock();
    double timeTaken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", timeTaken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_44_output.txt");
        FILE* outputFile = fopen(outputFileName, "w");
        if (outputFile != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(outputFile, "%.2f ", h_c[i]);
            }
            fclose(outputFile);
        }
    }

    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}