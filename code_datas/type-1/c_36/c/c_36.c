#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void logistic(unsigned int n, float a, float* x, float* z) {
    for (unsigned int myId = 0; myId < n; myId++) {
        z[myId] = a * x[myId] * (1 - x[myId]);
    }
}

int main(int argc, char* argv[]) {
    const unsigned int arraySize = 50000000; // 增大数组大小
    float* h_x = (float*)malloc(arraySize * sizeof(float));
    float* h_z = (float*)malloc(arraySize * sizeof(float));
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    clock_t start = clock();

    srand(42);
    for (unsigned int i = 0; i < arraySize; ++i) {
        h_x[i] = (float)(rand()) / RAND_MAX; 
    }

    // 增加迭代次数
    for (int iter = 0; iter < 100; iter++) {
        logistic(arraySize, 2.0f, h_x, h_z);
        // 简单的后处理以确保计算不被优化掉
        for (unsigned int i = 0; i < arraySize; i++) {
            h_x[i] = h_z[i];
        }
    }

    clock_t end = clock();
    double timeTaken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_36_output.txt");
        FILE* outputFile = fopen(outputFileName, "w");
        if (outputFile != NULL) {
            for (unsigned int i = 0; i < 1000; i++) {
                fprintf(outputFile, "%.6f ", h_z[i]);
            }
            fclose(outputFile);
        }
    }

    free(h_x);
    printf("Time: %.2f(ms)\n", timeTaken);
    free(h_z);
    return 0;
}