#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void doubleArrayVectorAdd(double* d_in_a, double* d_in_b, double* d_out, int length) {
    for (int idx = 0; idx < length; idx++) {
        d_out[idx] = d_in_a[idx] + d_in_b[idx];
    }
}

int main(int argc, char* argv[]) {
    const int arraySize = 10000000; // 增大数组大小
    double* h_d_in_a = (double*)malloc(arraySize * sizeof(double));
    double* h_d_in_b = (double*)malloc(arraySize * sizeof(double));
    double* h_d_out = (double*)malloc(arraySize * sizeof(double));
    
    for (int i = 0; i < arraySize; ++i) {
        h_d_in_a[i] = (double)(i);
        h_d_in_b[i] = (double)(2 * i);
    }

    clock_t start = clock();

    // 增加迭代次数以延长执行时间
    for (int iter = 0; iter < 1000; iter++) {
        doubleArrayVectorAdd(h_d_in_a, h_d_in_b, h_d_out, arraySize);
    }

    clock_t end = clock();
    double timeElapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", timeElapsed * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_46_output.txt");
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.2f ", h_d_out[i]);
            }
            fclose(file);
        }
    }

    free(h_d_in_a);
    free(h_d_in_b);
    free(h_d_out);
    return 0;
}