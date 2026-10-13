#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void multiplyIntValues(int* destination, int* vector, int value, unsigned int end) {
    for (unsigned int i = 0; i < end; i++) {
        destination[i] = vector[i] * value;
    }
}

int main(int argc, char* argv[]) {
    const int arraySize = 10000000;
    int* h_result = (int*)malloc(arraySize * sizeof(int));
    int* h_val1 = (int*)malloc(arraySize * sizeof(int));
    const int h_val2 = 2;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < arraySize; ++i) {
        h_val1[i] = i;
    }

    clock_t start = clock();

    multiplyIntValues(h_result, h_val1, h_val2, arraySize);

    clock_t end = clock();

    double timeElapsed = (double)(end - start) / CLOCKS_PER_SEC;

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_32_output.txt");
        FILE* outputFile = fopen(outputFileName, "w");
        if (outputFile != NULL) {
            for (unsigned int i = 0; i < 1000; i++) {
                fprintf(outputFile, "%d ", h_result[i]);
            }
            fclose(outputFile);
        }
    }

    free(h_result);
    free(h_val1);
    printf("Time: %.2fms\n", timeElapsed * 1000);
    return 0;
}