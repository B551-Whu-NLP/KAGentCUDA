#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void matrix_mult(int left_rows, int shared_dimensions, int right_columns, float* left, float* right, float* result) {
    int row, column, cell;
    for (row = 0; row < left_rows; row++) {
        for (column = 0; column < right_columns; column++) {
            result[row * right_columns + column] = 0;
            for (cell = 0; cell < shared_dimensions; cell++) {
                result[row * right_columns + column] += left[row * shared_dimensions + cell] * right[cell * right_columns + column];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int left_rows = 1000;      
    int shared_dimensions = 500; 
    int right_columns = 2000;   

    float* h_left = (float*)malloc(left_rows * shared_dimensions * sizeof(float));
    float* h_right = (float*)malloc(shared_dimensions * right_columns * sizeof(float));
    float* h_result = (float*)malloc(left_rows * right_columns * sizeof(float));

    for (int i = 0; i < left_rows * shared_dimensions; ++i) {
        h_left[i] = (float)(i % 100);
    }
    for (int i = 0; i < shared_dimensions * right_columns; ++i) {
        h_right[i] = (float)(i % 100); 
    }

    clock_t start = clock();

    matrix_mult(left_rows, shared_dimensions, right_columns, h_left, h_right, h_result);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_125_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {   
                for (int j = 0; j < 1000; j++) {
                    fprintf(fp, "h_result[%d * %d + %d] = %f\n", i, right_columns, j, h_result[i * right_columns + j]);
                }
            }
            fclose(fp);
        }
    }

    free(h_left);
    free(h_right);
    free(h_result);

    return 0;
}