#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void add(int n, float* x, float* y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main(int argc, char* argv[]) {
    int arraySize = 100000000; // 增大数组大小
    float* h_x = (float*)malloc(arraySize * sizeof(float));
    float* h_y = (float*)malloc(arraySize * sizeof(float));

    for (int i = 0; i < arraySize; ++i) {
        h_x[i] = i;
        h_y[i] = 2 * i;
    }

    clock_t start = clock();
    add(arraySize, h_x, h_y);
    clock_t end = clock();

    double timeTaken = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;

    printf("Time: %.2f(ms)\n", timeTaken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE* file = fopen("c_4_output.txt", "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.2f ", h_y[i]);
            }
            fclose(file);
        }
    }

    free(h_x);
    free(h_y);
    return 0;
}