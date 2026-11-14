#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int dim_rows = 1000;
    int dim_cols = 1500;
    int* h_in_mat = (int*)malloc(dim_rows * dim_cols * sizeof(int));
    int* h_out_mat = (int*)malloc(dim_cols * dim_rows * sizeof(int));

    for (int i = 0; i < dim_rows * dim_cols; ++i) {
        h_in_mat[i] = i;
    }

    clock_t start = clock();

    for (int iter = 0; iter < 100; ++iter) {
        for (int i = 0; i < dim_rows; ++i) {
            for (int j = 0; j < dim_cols; ++j) {
                unsigned int new_pos = j * dim_rows + i;
                h_out_mat[new_pos] = h_in_mat[i * dim_cols + j];
            }
        }
    }

    clock_t end = clock();
    
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_281_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                for (int j = 0; j < 1000; ++j) {
                    fprintf(fp, "%d ", h_out_mat[i * dim_rows + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_in_mat);
    free(h_out_mat);

    return 0;
}