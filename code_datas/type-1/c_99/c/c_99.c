#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void globalCalculateKernel(float *c, float *a, float *b, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            c[i * size + j] = sinf(a[i * size + j]) * sinf(a[i * size + j]) +
                               cosf(b[i * size + j]) * cosf(b[i * size + j]) * cosf(b[i * size + j]);
        }
    }
}

int main(int argc, char *argv[]) {
    int size = 100;
    size_t bytes = size * size * sizeof(float);
    float* h_a, * h_b, * h_c;
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    for (int i = 0; i < size * size; i++) {
        h_a[i] = (float)(i % 100);
        h_b[i] = (float)(i % 100);
    }

    clock_t start = clock();

    globalCalculateKernel(h_c, h_a, h_b, size);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_99_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    fprintf(fp, "%.6f ", h_c[i * size + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_c);
    free(h_a);
    free(h_b);
    return 0;
}