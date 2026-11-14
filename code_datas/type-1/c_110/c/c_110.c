#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void fabsf_clamp(int N, float* X, int INCX, float clamp_min, float clamp_max) {
    for (int i = 0; i < N; ++i) {
        if (X[i * INCX] >= 0) {
            X[i * INCX] = fminf(clamp_max, fmaxf(clamp_min, X[i * INCX]));
        }
        else {
            X[i * INCX] = fminf(-clamp_min, fmaxf(-clamp_max, X[i * INCX]));
        }
    }
}

int main(int argc, char* argv[]) {
    int N = 1024;
    float clamp_min = -1.0f;
    float clamp_max = 1.0f;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* h_X = (float*)malloc(N * sizeof(float));

    srand(42);
    for (int i = 0; i < N; i++) {
        h_X[i] = ((float)rand() / RAND_MAX) * 100.0f - 50.0f;
    }

    clock_t start_time = clock();

    fabsf_clamp(N, h_X, 1, clamp_min, clamp_max);

    clock_t end_time = clock();

    double elapsed_time = ((double)(end_time - start_time) * 1000.0) / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", elapsed_time);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_110_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < N; i++) {
                fprintf(fp, "%.6f ", h_X[i]);
            }
            fclose(fp);
        }
    }

    free(h_X);
    return 0;
}