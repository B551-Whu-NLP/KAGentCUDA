#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void addMatrix(float *a, float *b, float *c, int N) {
    int i, j, idx;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            idx = i * N + j;
            a[idx] = b[idx] + c[idx];
        }
    }
}

int main(int argc, char *argv[]) {
    int N = 1000;
    float* h_a = (float*)malloc(N * N * sizeof(float));
    float* h_b = (float*)malloc(N * N * sizeof(float));
    float* h_c = (float*)malloc(N * N * sizeof(float));
    
    for (int i = 0; i < N * N; ++i) {
        h_a[i] = (float)(i + 1);
        h_b[i] = (float)(i + 2);
        h_c[i] = (float)(i + 3);
    }

    clock_t start = clock();

    addMatrix(h_a, h_b, h_c, N);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_77_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 100; ++i) {
                for (int j = 0; j < 100; ++j) {
                    fprintf(fp, "%.2f ", h_a[i * N + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    printf("Time: %.2fms\n", time_taken * 1000);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}