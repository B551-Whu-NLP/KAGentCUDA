#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

static inline void MulMatrix(float* A, float* B, float* C, int nx, int ny) {
    int i, j, k;
    float sum = 0.0;
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            sum = 0.0;
            for (k = 0; k < nx; k++) {
                sum = sum + A[i * nx + k] * B[k * ny + j];
            }
            C[i * ny + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int nx = 1024;
    int ny = 512;
    float* h_A = (float*)malloc(nx * nx * sizeof(float));
    float* h_B = (float*)malloc(nx * ny * sizeof(float));
    float* h_C = (float*)malloc(nx * ny * sizeof(float));

    for (int i = 0; i < nx * nx; i++) {
        h_A[i] = (float)(i % 100);
    }
    for (int i = 0; i < nx * ny; i++) {
        h_B[i] = (float)(i % 100);
    }

    clock_t start = clock();
    MulMatrix(h_A, h_B, h_C, nx, ny);
    clock_t end = clock();
    double time_taken = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;

    printf("Time: %.4f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_112_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                for (int j = 0; j < 1000; j++) {
                    fprintf(fp, "%.4f ", h_C[i * ny + j]);
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