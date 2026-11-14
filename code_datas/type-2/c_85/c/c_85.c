#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void matrix_transpose(int *mat_in, int *mat_out, unsigned int rows, unsigned int cols) {
    unsigned int idx;
    unsigned int idy;

    clock_t start = clock();

    for (idx = 0; idx < cols; idx++) {
        for (idy = 0; idy < rows; idy++) {
            unsigned int pos = idy * cols + idx;
            unsigned int trans_pos = idx * rows + idy;
            mat_out[trans_pos] = mat_in[pos];
        }
    }

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time: %.2f\n", time_taken);
}

int main(int argc, char *argv[]) {
    unsigned int rows = 1000;
    unsigned int cols = 1200;
    size_t size = rows * cols * sizeof(int);
    int* h_mat_in = (int*)malloc(size);
    int* h_mat_out = (int*)malloc(size);
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (unsigned int i = 0; i < rows * cols; ++i) {
        h_mat_in[i] = i;
    }

    matrix_transpose(h_mat_in, h_mat_out, rows, cols);

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "%s_output.txt", argv[0]);
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (unsigned int i = 0; i < cols; ++i) {
                for (unsigned int j = 0; j < rows; ++j) {
                    fprintf(fp, "%d ", h_mat_out[i * rows + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_mat_in);
    free(h_mat_out);

    return 0;
}