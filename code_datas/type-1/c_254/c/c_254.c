#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void axpy(int N, float ALPHA, float* X, int OFFX, int INCX, float* Y, int OFFY, int INCY) {
    for (int i = 0; i < N; ++i) {
        Y[OFFY + i * INCY] += ALPHA * X[OFFX + i * INCX];
    }
}

int main(int argc, char* argv[]) {
    int N = 1000000;
    float ALPHA = 2.0f;
    float* h_X = (float*)malloc(N * sizeof(float));
    float* h_Y = (float*)malloc(N * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    srand(42);
    for (int i = 0; i < N; ++i) {
        h_X[i] = (float)(rand() % 100);
        h_Y[i] = (float)(rand() % 100);
    }

    clock_t start = clock();
    for (int iter = 0; iter < 1000; ++iter) {
        axpy(N, ALPHA, h_X, 0, 1, h_Y, 0, 1);
    }
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_254_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", h_Y[i]);
            }
            fclose(fp);
        }
    }

    free(h_X);
    printf("Time: %.2f(ms)\n", time_taken);
    free(h_Y);
    return 0;
}