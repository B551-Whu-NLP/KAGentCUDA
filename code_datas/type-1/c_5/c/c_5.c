#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void scale(float* array, float scale, int N) {
    for (int idx = 0; idx < N; idx++) {
        array[idx] *= scale;
    }
}

int main(int argc, char** argv) {
    int arraySize = 10000000; // 增大数组大小
    float scaleFactor = 1.5f;
    float* h_array = (float*)malloc(arraySize * sizeof(float));

    for (int i = 0; i < arraySize; ++i) {
        h_array[i] = i;
    }

    clock_t start = clock();

    scale(h_array, scaleFactor, arraySize);

    clock_t end = clock();
    double timeTaken = (double)(end - start) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[] = "c_5_output.txt";
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.2f ", h_array[i]);
            }
            fprintf(file, "\nTime: %.2fms", timeTaken * 1000);
            fclose(file);
        }
    }

    free(h_array);
    return 0;
}