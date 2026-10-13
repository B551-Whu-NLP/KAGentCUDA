#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void matrix_mult(int* h_a, int* h_b, int* h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            int tmp = 0;
            for (int h = 0; h < n; ++h) {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

int main(int argc, char** argv) {
    int m = 2048;
    int n = 1024;
    int k = 512;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    srand(42);

    int* h_a = (int*)malloc(m * n * sizeof(int));
    int* h_b = (int*)malloc(n * k * sizeof(int));
    int* h_c = (int*)malloc(m * k * sizeof(int));

    for (int i = 0; i < m * n; i++) {
        h_a[i] = rand() % 10;
    }

    for (int i = 0; i < n * k; i++) {
        h_b[i] = rand() % 10;
    }

    clock_t start = clock();

    matrix_mult(h_a, h_b, h_c, m, n, k);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Time: %.2fms\n", time_taken * 1000);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_111_output.txt");

        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                for (int j = 0; j < 1000; j++) {
                    fprintf(fp, "%d ", h_c[i * k + j]);
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