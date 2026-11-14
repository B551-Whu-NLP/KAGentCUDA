#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void mat_mul(int* m_A, int* m_B, int* m_C, int A_rows, int A_cols, int B_rows, int B_cols, int verbose) {
    clock_t start = clock();
    
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            int sum = 0;
            for (int k = 0; k < A_cols; k++) {
                sum += m_A[i * A_cols + k] * m_B[k * B_cols + j];
            }
            m_C[i * B_cols + j] = sum;
        }
    }
    
    clock_t end = clock();
    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_spent * 1000);
    
    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_320_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < A_rows; i++) {
                for (int j = 0; j < B_cols; j++) {
                    fprintf(fp, "%d ", m_C[i * B_cols + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }
}

int main(int argc, char *argv[]) {
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    
    int A_rows = 100, A_cols = 120;
    int B_rows = 120, B_cols = 150;

    int *matrix_A = (int*)malloc(A_rows * A_cols * sizeof(int));
    int *matrix_B = (int*)malloc(B_rows * B_cols * sizeof(int));
    int *matrix_C = (int*)malloc(A_rows * B_cols * sizeof(int));

    for (int i = 0; i < A_rows * A_cols; i++) {
        matrix_A[i] = i % 100 + 1;
    }
    for (int i = 0; i < B_rows * B_cols; i++) {
        matrix_B[i] = (i * 2) % 100 + 1;
    }

    mat_mul(matrix_A, matrix_B, matrix_C, A_rows, A_cols, B_rows, B_cols, verbose);

    free(matrix_A);
    free(matrix_B);
    free(matrix_C);
    return 0;
}