#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void copyAliasRow(int *devMat, int memWidth, int memHeight, double *timeElapsed) {
    clock_t start = clock();

    for (int iter = 0; iter < 1000; iter++) {
        for (int devMatX = 1; devMatX < memWidth; devMatX++) {
            devMat[memWidth * 0 + devMatX] = devMat[memWidth * (memHeight - 2) + devMatX];
            devMat[memWidth * (memHeight - 1) + devMatX] = devMat[memWidth * 1 + devMatX];
        }
    }

    clock_t end = clock();
    *timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
}

int main(int argc, char *argv[]) {
    int memWidth = 5000;
    int memHeight = 5000;
    int* devMat = (int*)malloc(memWidth * memHeight * sizeof(int));

    for (int i = 0; i < memWidth * memHeight; i++) {
        devMat[i] = i;
    }

    double timeElapsed;
    copyAliasRow(devMat, memWidth, memHeight, &timeElapsed);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_90_output.txt");
        FILE *file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 100; i++) {
                for (int j = 0; j < 100; j++) {
                    fprintf(file, "%d ", devMat[i * memWidth + j]);
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    printf("Time: %.2f(ms)\n", timeElapsed);
    free(devMat);
    return 0;
}