#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void matrix_mul(int *a, int *b, int *c, int N) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            int sum = 0;
            for (int i = 0; i < N; i++) {
                sum += a[row * N + i] * b[i * N + col];
            }
            c[row * N + col] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int N = 1000;
    int* h_a = (int*)malloc(N * N * sizeof(int));
    int* h_b = (int*)malloc(N * N * sizeof(int));
    int* h_c = (int*)malloc(N * N * sizeof(int));

    for (int i = 0; i < N * N; i++) {
        h_a[i] = 1;
        h_b[i] = 2;
    }

    clock_t start = clock();
    matrix_mul(h_a, h_b, h_c, N);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_1000_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    fprintf(fp, "%d ", h_c[i * N + j]);
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