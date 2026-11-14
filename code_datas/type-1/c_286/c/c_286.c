#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void smallCorrelation(float* L, float* innerSums, int innerSumsLength, double* timeTaken) {
    clock_t start = clock();

    for (int u = 0; u < innerSumsLength; u++) {
        int realIdx = 2 * u;
        int imagIdx = realIdx + 1;
        L[u] = (innerSums[realIdx] * innerSums[realIdx]) + (innerSums[imagIdx] * innerSums[imagIdx]);
    }

    clock_t end = clock();
    *timeTaken = ((double)(end - start)) / CLOCKS_PER_SEC;
}

int main(int argc, char** argv) {
    const int innerSumsLength = 1000000; // 增加1000倍
    float* h_L = (float*)malloc(innerSumsLength * sizeof(float));
    float* h_innerSums = (float*)malloc(2 * innerSumsLength * sizeof(float));

    for (int i = 0; i < 2 * innerSumsLength; ++i) {
        h_innerSums[i] = i;
    }

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    double timeTaken;
    smallCorrelation(h_L, h_innerSums, innerSumsLength, &timeTaken);

    if (verbose) {
        FILE* file = fopen("c_286_output.txt", "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f\t", h_L[i]);
            }
            fclose(file);
        }
    }

    free(h_L);
    free(h_innerSums);

    printf("Time: %.2fms\n", timeTaken * 1000);

    return 0;
}