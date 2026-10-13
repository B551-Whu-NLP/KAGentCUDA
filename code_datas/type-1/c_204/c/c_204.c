#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void add_matrix(float* a, float* b, float* c, int N) {
    for (int idx = 0; idx < N; idx++) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char* argv[]) {
    const int matrix_size = 1000000; // 增大矩阵尺寸
    float* a_host = (float*)malloc(matrix_size * sizeof(float));
    float* b_host = (float*)malloc(matrix_size * sizeof(float));
    float* c_result = (float*)malloc(matrix_size * sizeof(float));

    for (int i = 0; i < matrix_size; ++i) {
        a_host[i] = i + 1.0;
        b_host[i] = (i + 1.0) * 2.0;
    }

    clock_t start = clock();

    // 增加迭代次数以延长执行时间
    for (int iter = 0; iter < 1000; iter++) {
        add_matrix(a_host, b_host, c_result, matrix_size);
    }

    clock_t end = clock();

    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_spent * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE* fp = fopen("c_204_output.txt", "w");
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

    return 0;
}