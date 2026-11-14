#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void iKernel(float* A, float* B, float* C, const int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char* argv[]) {
    const int arraySize = 10000000; // 增大数组大小
    float* h_A = (float*)malloc(arraySize * sizeof(float));
    float* h_B = (float*)malloc(arraySize * sizeof(float));
    float* h_C = (float*)malloc(arraySize * sizeof(float));

    for (int i = 0; i < arraySize; ++i) {
        h_A[i] = (float)(i);
        h_B[i] = (float)(2 * i);
    }

    clock_t start, end;
    double cpu_time_used;

    start = clock();
    iKernel(h_A, h_B, h_C, arraySize);
    end = clock();

    cpu_time_used = ((double) (end - start)) * 1000 / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_31_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.2f ", h_C[i]);
            }
            fclose(file);
        }
    }

    free(h_A);
    free(h_B);
    free(h_C);
    printf("Time: %.2f(ms)\n", cpu_time_used);
    return 0;
}