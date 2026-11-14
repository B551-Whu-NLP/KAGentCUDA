#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void add_matrix_cpu(double* a, double* b, double* c, int size) {
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char *argv[]) {
    const int matrix_size = 1000000; // 增大矩阵大小
    double* a_host = (double*)malloc(matrix_size * sizeof(double));
    double* b_host = (double*)malloc(matrix_size * sizeof(double));
    double* c_result = (double*)malloc(matrix_size * sizeof(double));

    for (int i = 0; i < matrix_size; ++i) {
        a_host[i] = i + 1.0;
        b_host[i] = (i + 1.0) * 2.0;
    }

    clock_t start_time = clock();

    // 增加迭代次数以延长执行时间
    for (int iter = 0; iter < 1000; ++iter) {
        add_matrix_cpu(a_host, b_host, c_result, matrix_size);
    }

    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE *fp = fopen("c_207_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", c_result[i]);
            }
            fclose(fp);
        }
    }

    free(a_host);
    free(b_host);
    free(c_result);
    
    printf("Time: %.2f(ms)\n", elapsed_time * 1000);

    return 0;
}