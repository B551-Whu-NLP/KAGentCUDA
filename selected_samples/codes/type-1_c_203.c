#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void subtract_matrix(float* a, float* b, float* c, int N) {
    for (int idx = 0; idx < N; idx++) {
        c[idx] = a[idx] - b[idx];
    }
}

int main(int argc, char* argv[]) {
    const int matrix_size = 1000000;
    float* a_host = (float*)malloc(matrix_size * sizeof(float));
    float* b_host = (float*)malloc(matrix_size * sizeof(float));
    float* c_result = (float*)malloc(matrix_size * sizeof(float));

    for (int i = 0; i < matrix_size; ++i) {
        a_host[i] = i + 1.0;
        b_host[i] = (i + 1.0) * 2.0;
    }

    clock_t start = clock();

    subtract_matrix(a_host, b_host, c_result, matrix_size);

    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time: %.2f\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_203_output.txt");
        FILE* fp = fopen(output_file, "w");
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