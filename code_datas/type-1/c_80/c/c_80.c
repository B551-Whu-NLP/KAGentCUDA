#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void AddMatrixOn(float* A, float* B, float* C, int nx, int ny) {
    int i, j;
    int cnt = 0;
    for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
            C[cnt] = A[cnt] + B[cnt];
            cnt++;
        }
    }
}

int main(int argc, char *argv[]) {
    int nx = 5000;
    int ny = 5000;
    float* h_A = (float*)malloc(nx * ny * sizeof(float));
    float* h_B = (float*)malloc(nx * ny * sizeof(float));
    float* h_C = (float*)malloc(nx * ny * sizeof(float));

    for (int i = 0; i < nx * ny; ++i) {
        h_A[i] = (float)(i + 1);
        h_B[i] = (float)(i + 2);
    }

    clock_t start = clock();

    for(int iter = 0; iter < 100; iter++) {
        AddMatrixOn(h_A, h_B, h_C, nx, ny);
    }

    clock_t end = clock();
    double time_taken = (double)(end - start) * 1000 / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE *fp = fopen("c_80_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 100; ++i) {
                for (int j = 0; j < 100; ++j) {
                    fprintf(fp, "%.2f ", h_C[i * ny + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}