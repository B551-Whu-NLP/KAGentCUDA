#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void matrixProduct(double* matrix_a, double* matrix_b, double* matrix_c, int width, int height, int from, int my_rank, int verbose) {
    int row, col;

    clock_t start = clock();

    for (int iter = 0; iter < 100; iter++) {
        for (row = 0; row < width; row++) {
            for (col = 0; col < height; col++) {
                matrix_c[row * width + col] = 0;
                for (int k = 0; k < width; k++) {
                    matrix_c[row * width + col] += matrix_a[((row + from) * width) + k] * matrix_b[k * width + col];
                }
            }
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        char output_file[256];
        sprintf(output_file, "c_307_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    fprintf(fp, "%.2f ", matrix_c[i * width + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }
}

int main(int argc, char *argv[]) {
    int width = 500;
    int height = 500;
    int from = 0;
    int my_rank = 0;
    int verbose = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        }
    }

    double *matrix_a = (double *)malloc(width * height * sizeof(double));
    double *matrix_b = (double *)malloc(width * height * sizeof(double));
    double *matrix_c = (double *)malloc(width * height * sizeof(double));

    srand(42);

    for (int i = 0; i < width * height; i++) {
        matrix_a[i] = (double)(rand() % 100);
        matrix_b[i] = (double)(rand() % 100); 
    }

    matrixProduct(matrix_a, matrix_b, matrix_c, width, height, from, my_rank, verbose);

    free(matrix_a);
    free(matrix_b);
    free(matrix_c);

    return 0;
}