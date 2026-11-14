#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void add03(float* c, float* a, float* b, int n) {
    for (int k = 0; k < n; k++) {
        c[k] = a[k] + b[k];
    }
}

int main(int argc, char* argv[]) {
    int arraySize = 10000000; // 增大数组大小
    float* h_a = (float*)malloc(arraySize * sizeof(float));
    float* h_b = (float*)malloc(arraySize * sizeof(float));
    float* h_c = (float*)malloc(arraySize * sizeof(float));

    for (int i = 0; i < arraySize; ++i) {
        h_a[i] = (float)(i);
        h_b[i] = (float)(2 * i);
    }

    clock_t start = clock();

    // 增加循环次数以延长执行时间
    for (int iter = 0; iter < 100; iter++) {
        add03(h_c, h_a, h_b, arraySize);
    }

    clock_t end = clock();
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", timeElapsed * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_18_output.txt");
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