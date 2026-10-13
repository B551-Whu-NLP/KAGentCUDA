#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int matrixMul(float* h_M, float* h_N, float* h_P, int width) {
    int Pvalue;
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            Pvalue = 0;
            for (int k = 0; k < width; ++k) {
                Pvalue += h_M[row * width + k] * h_N[k * width + col];
            }
            h_P[row * width + col] = Pvalue;
        }
    }
    return 0;
}

int main(int argc, char *argv[]) {
    int width = 1000;
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* h_M = (float*)malloc(width * width * sizeof(float));
    float* h_N = (float*)malloc(width * width * sizeof(float));
    float* h_P = (float*)malloc(width * width * sizeof(float));

    srand(0);
    for (int i = 0; i < width * width; i++) {
        h_M[i] = (float)(rand() % 100);
    }
    for (int i = 0; i < width * width; i++) {
        h_N[i] = (float)(rand() % 100);
    }

    clock_t start = clock();

    matrixMul(h_M, h_N, h_P, width);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        char output_filename[256];
        sprintf(output_filename, "%s_output.txt", "c_113");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                for (int j = 0; j < 1000; j++) {
                    fprintf(fp, "%.2f ", h_P[i * width + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_M);
    free(h_N);
    free(h_P);
    return 0;
}