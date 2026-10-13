#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void add_vector(float* a, float* b, float* c, int size) {
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv) {
    int arraySize = 100000000; // 增大数组大小
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* h_arrayA = (float*)malloc(arraySize * sizeof(float));
    float* h_arrayB = (float*)malloc(arraySize * sizeof(float));
    float* h_output = (float*)malloc(arraySize * sizeof(float));

    for (int i = 0; i < arraySize; ++i) {
        h_arrayA[i] = (float)(i);
        h_arrayB[i] = (float)(2 * i);
    }

    clock_t start = clock();

    add_vector(h_arrayA, h_arrayB, h_output, arraySize);

    clock_t end = clock();
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", timeElapsed * 1000);

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_9_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.2f ", h_output[i]);
            }
            fclose(file);
        }
    }

    free(h_arrayA);
    free(h_arrayB);
    free(h_output);
    return 0;
}