#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void saxpy(const int dim, float a, float* x, float* y) {
    for (int i = 0; i < dim; i++) {
        y[i] += a * x[i];
    }
}

int main(int argc, char* argv[]) {
    int arraySize = 100000000;  // 增大数组大小10倍
    float a = 2.0;
    float* h_x = (float*)malloc(arraySize * sizeof(float));
    float* h_y = (float*)malloc(arraySize * sizeof(float));

    for (int i = 0; i < arraySize; ++i) {
        h_x[i] = (float)(i);
        h_y[i] = (float)(2 * i);
    }

    clock_t start = clock();

    // 增加循环次数，重复执行saxpy操作
    for (int iter = 0; iter < 100; iter++) {
        saxpy(arraySize, a, h_x, h_y);
    }

    clock_t end = clock();

    double timeElapsed = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;
    printf("Time: %.2f(ms)\n", timeElapsed);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE* file = fopen("c_22_output.txt", "w");
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