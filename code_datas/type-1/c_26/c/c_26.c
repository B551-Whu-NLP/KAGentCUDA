#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int arraySize = 1000000; // 增大数组大小
    double value = 2.0;
    double* h_a = (double*)malloc(arraySize * sizeof(double));
    double* h_c = (double*)malloc(arraySize * sizeof(double));
    for (int i = 0; i < arraySize; ++i) {
        h_a[i] = (double)(i);
    }

    clock_t start = clock();

    void dmul_Scalar_matrix(double* a, double value, double* c, int N) {
        for (int idx = 0; idx < N; idx++) {
            c[idx] = a[idx] * value;
        }
    }

    dmul_Scalar_matrix(h_a, value, h_c, arraySize);

    clock_t end = clock();
    double time_taken = (double)(end - start) * 1000 / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE *fp = fopen("c_26_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", h_c[i]);
            }
            fclose(fp);
        }
    }

    free(h_a);
    free(h_c);
    return 0;
}