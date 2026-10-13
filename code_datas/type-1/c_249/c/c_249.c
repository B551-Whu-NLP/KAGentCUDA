#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void gaussianPass(int patchSize, int dataSize, float* gaussFilter, float* data) {
    for (int i = 0; i < dataSize; i++) {
        data[i] = gaussFilter[i % (patchSize * patchSize)] * data[i];
    }
}

int main(int argc, char* argv[]) {
    int dataSize = 1000000;
    int patchSize = 500;
    float* h_gaussFilter = (float*)malloc(patchSize * patchSize * sizeof(float));
    float* h_data = (float*)malloc(dataSize * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < patchSize * patchSize; ++i) {
        h_gaussFilter[i] = (float)(i + 1);
    }
    for (int i = 0; i < dataSize; ++i) {
        h_data[i] = (float)(i + 1);
    }

    clock_t start = clock();

    gaussianPass(patchSize, dataSize, h_gaussFilter, h_data);

    clock_t end = clock();
    double timeTaken = ((double)(end - start)) / CLOCKS_PER_SEC;
    long long timeInMs = (long long)(timeTaken * 1000);

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_249_output.txt");
        FILE* outputFile = fopen(outputFileName, "w");
        if (outputFile != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(outputFile, "%.2f ", h_data[i]);
            }
            fclose(outputFile);
        }
    }

    free(h_gaussFilter);
    free(h_data);
    printf("Time: %lld(ms)\n", timeInMs);

    return 0;
}