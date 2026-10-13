#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void vecAdd(int n, double* pdbA, double* pdbB, double* pdbC) {
    for (int i = 0; i < n; ++i) {
        pdbC[i] = pdbA[i] + pdbB[i];
    }
}

int main(int argc, char* argv[]) {
    const int vector_size = 10000000;
    double* pdbA_host = (double*)malloc(vector_size * sizeof(double));
    double* pdbB_host = (double*)malloc(vector_size * sizeof(double));
    double* pdbC_result = (double*)malloc(vector_size * sizeof(double));

    for (int i = 0; i < vector_size; ++i) {
        pdbA_host[i] = i + 1.0;
        pdbB_host[i] = (i + 1.0) * 2.0;
    }

    clock_t start = clock();

    vecAdd(vector_size, pdbA_host, pdbB_host, pdbC_result);

    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE* file = fopen("c_209_output.txt", "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", pdbC_result[i]);
            }
            fclose(file);
        }
    }

    free(pdbA_host);
    free(pdbB_host);
    free(pdbC_result);

    printf("Time: %.2f\n", time_taken);

    return 0;
}