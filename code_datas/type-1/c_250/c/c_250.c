#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void add(int* a, int* b, int* c, int vectorSize, double* timeElapsed) {
    clock_t start = clock();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < vectorSize; i++) {
        c[i] = a[i] + b[i];
    }

    clock_t end = clock();
    *timeElapsed = ((double)(end - start)) * 1000.0 / CLOCKS_PER_SEC;
}

int main(int argc, char* argv[]) {
    int vectorSize = 25000000; // 增大向量大小
    int* h_a = (int*)malloc(vectorSize * sizeof(int));
    int* h_b = (int*)malloc(vectorSize * sizeof(int));
    int* h_c = (int*)malloc(vectorSize * sizeof(int));
    int verbose = 0;
    double timeElapsed;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < vectorSize; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    add(h_a, h_b, h_c, vectorSize, &timeElapsed);

    if (verbose) {
        char outputFileName[256];
        sprintf(outputFileName, "%s_output.txt", argv[0]);
        FILE* file = fopen(outputFileName, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%d ", h_c[i]);
            }
            fclose(file);
        }
    }

    free(h_a);
    free(h_b);
    free(h_c);

    printf("Time: %.2f(ms)\n", timeElapsed);
    return 0;
}