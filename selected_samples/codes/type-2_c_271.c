#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void mean(float* A, float* means, int size_row, int size_col) {
    for (int idx = 0; idx < size_col; idx++) {
        for (int i = 0; i < size_row; i++) {
            means[idx] += A[idx * size_row + i];
        }
        means[idx] = means[idx] / size_row;
    }
}

int main(int argc, char *argv[]) {
    int size_row = 10000;
    int size_col = 10000;
    float* h_A = (float*)malloc(size_row * size_col * sizeof(float));
    float* h_means = (float*)calloc(size_col, sizeof(float));

    clock_t start = clock();

    for (int i = 0; i < size_row * size_col; ++i) {
        h_A[i] = i + 1;
    }

    mean(h_A, h_means, size_row, size_col);

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_271_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", h_A[i]);
            }
            fprintf(fp, "\n");
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "Column %d: %.2f\n", i, h_means[i]);
            }
            fclose(fp);
        }
    }

    free(h_A);
    free(h_means);

    return 0;
}