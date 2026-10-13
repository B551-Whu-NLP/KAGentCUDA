#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void fill_matrix(double* const A, const int rows, const int cols) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            A[row * cols + col] = row;
        }
    }
}

int main(int argc, char *argv[]) {
    const int rows = 10000;
    const int cols = 5000;
    double* h_A = (double*)malloc(rows * cols * sizeof(double));
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    clock_t start = clock();

    fill_matrix(h_A, rows, cols);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_47_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int row = 0; row < 1000; row++) {
                for (int col = 0; col < 1000; col++) {
                    fprintf(fp, "%.2f ", h_A[row * cols + col]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    printf("Time: %.2fms\n", time_taken * 1000);
    free(h_A);
    return 0;
}