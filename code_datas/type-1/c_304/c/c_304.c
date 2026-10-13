#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void inter(int NX, float* X, int NY, float* Y, int B, float* OUT) {
    int i, j;
    int index = 0;
    for (j = 0; j < B; ++j) {
        for (i = 0; i < NX; ++i) {
            OUT[index++] = X[j * NX + i];
        }
        for (i = 0; i < NY; ++i) {
            OUT[index++] = Y[j * NY + i];
        }
    }
}

int main(int argc, char* argv[]) {
    int NX = 3000;
    int NY = 2000;
    int B = 4000;
    float* X = (float*)malloc(NX * B * sizeof(float));
    float* Y = (float*)malloc(NY * B * sizeof(float));
    float* OUT = (float*)malloc((NX + NY) * B * sizeof(float));
    
    for (int i = 0; i < NX * B; i++) {
        X[i] = (float)(i + 1);
    }
    for (int i = 0; i < NY * B; i++) {
        Y[i] = 0.1 * (float)(i + 1);
    }

    clock_t start = clock();

    inter(NX, X, NY, Y, B, OUT);

    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_304_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.2f ", OUT[i]);
            }
            fclose(file);
        }
    }

    free(X);
    free(Y);
    free(OUT);
    printf("Time: %.2f(ms)\n", time_taken * 1000);
    return 0;
}