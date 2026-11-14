#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void mmul(const float* A, const float* B, float* C, int r1, int c1, int r2, int c2, int verbose) {
    clock_t start = clock();

    for (int iter = 0; iter < 1000; iter++) {
        for (int idx = 0; idx < c2; idx++) {
            for (int idy = 0; idy < r1; idy++) {
                float temp = 0;
                for (int i = 0; i < c1; i++) {
                    temp += A[idy * c1 + i] * B[i * c2 + idx];
                }
                C[idy * c2 + idx] = temp;
            }
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);

    if (verbose) {
        char output_file[256];
        snprintf(output_file, sizeof(output_file), "c_114_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            int print_rows = (r1 < 100) ? r1 : 100;
            int print_cols = (c2 < 100) ? c2 : 100;
            for (int i = 0; i < print_rows; i++) {
                for (int j = 0; j < print_cols; j++) {
                    fprintf(fp, "%.2f ", C[i * c2 + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }
}

int main(int argc, char* argv[]) {
    int r1 = 160;
    int c1 = 80;
    int r2 = 80;
    int c2 = 40;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* h_A = (float*)malloc(r1 * c1 * sizeof(float));
    float* h_B = (float*)malloc(r2 * c2 * sizeof(float));
    float* h_C = (float*)malloc(r1 * c2 * sizeof(float));

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    srand(0);
    for (int i = 0; i < r1 * c1; i++) {
        h_A[i] = (float)(rand() % 100);
    }
    for (int i = 0; i < r2 * c2; i++) {
        h_B[i] = (float)(rand() % 100);
    }

    mmul(h_A, h_B, h_C, r1, c1, r2, c2, verbose);

    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
