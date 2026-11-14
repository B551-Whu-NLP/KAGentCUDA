#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

double clip(int N, float ALPHA, float* X, int INCX, float* Y, int INCY) {
    clock_t start = clock();

    for (int i = 0; i < N; ++i) {
        float val = X[i * INCX];
        Y[i * INCY] = val > ALPHA ? val : 0;
    }

    clock_t end = clock();
    return ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
}

int main(int argc, char** argv) {
    int N = 1000000;
    float ALPHA = 5.0;
    float* h_X = (float*)malloc(N * sizeof(float));
    float* h_Y = (float*)malloc(N * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < N; ++i) {
        h_X[i] = i - N / 2.0;
    }

    double time_taken = clip(N, ALPHA, h_X, 1, h_Y, 1);

    if (verbose) {
        char output_file[256];
        sprintf(output_file, "c_263_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", h_X[i]);
            }
            fprintf(file, "\nClipped Array (ALPHA=%.2f):\n", ALPHA);
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", h_Y[i]);
            }
            fclose(file);
        }
    }
    free(h_X);
    free(h_Y);

    printf("Time: %.2f(ms)\n", time_taken);
    return 0;
}