#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void initWith(float num, float* a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = num;
    }
}

int main(int argc, char* argv[]) {
    int arraySize = 100000000;  // 增加10倍
    float initialValue = 3.0;
    float* h_a = (float*)malloc(arraySize * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    clock_t start = clock();

    initWith(initialValue, h_a, arraySize);

    clock_t end = clock();
    double timeElapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", timeElapsed * 1000);

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_20_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.2f ", h_a[i]);
            }
            fclose(file);
        }
    }

    free(h_a);
    return 0;
}