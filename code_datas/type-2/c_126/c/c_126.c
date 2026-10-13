#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void matrixMultiplication(int* host_a, int* host_b, int* host_c, int row_a, int col_a, int col_b) {
    for (int i = 0; i < row_a; ++i) {
        for (int j = 0; j < col_b; ++j) {
            int tmp = 0;
            for (int k = 0; k < col_a; ++k) {
                tmp += host_a[i * col_a + k] * host_b[k * col_b + j];
            }
            host_c[i * col_b + j] = tmp;
        }
    }
}

int main(int argc, char *argv[]) {
    int row_a = 500;
    int col_a = 300;
    int col_b = 400;

    int* h_a = (int*)malloc(row_a * col_a * sizeof(int));
    int* h_b = (int*)malloc(col_a * col_b * sizeof(int));
    int* h_c = (int*)malloc(row_a * col_b * sizeof(int));

    for (int i = 0; i < row_a * col_a; i++) {
        h_a[i] = i;
    }
    for (int i = 0; i < col_a * col_b; i++) {
        h_b[i] = 2 * i;
    }

    clock_t start = clock();

    for (int iter = 0; iter < 100; iter++) {
        matrixMultiplication(h_a, h_b, h_c, row_a, col_a, col_b);
    }

    clock_t end = clock();

    double time_taken = (double)(end - start) * 1000 / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_126_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < row_a; i++) {
                for (int j = 0; j < col_b; j++) {
                    fprintf(fp, "%d ", h_c[i * col_b + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}