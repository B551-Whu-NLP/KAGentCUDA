#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void binarize_weights(float* weights, int n, int size, float* binary) {
    clock_t start = clock();

    for (int f = 0; f < n; ++f) {
        float mean = 0;

        for (int i = 0; i < size; ++i) {
            mean += fabs(weights[f * size + i]);
        }

        mean /= size;

        for (int i = 0; i < size; ++i) {
            binary[f * size + i] = (weights[f * size + i] > 0) ? mean : -mean;
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);
}

int main(int argc, char *argv[]) {
    int n = 1000;
    int size = 500;

    float* h_weights = (float*)malloc(n * size * sizeof(float));
    float* h_binary = (float*)malloc(n * size * sizeof(float));

    for (int i = 0; i < n * size; ++i) {
        h_weights[i] = (float)(i % 100);
    }

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    binarize_weights(h_weights, n, size, h_binary);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_123_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 100; i++) {
                for (int j = 0; j < 50; ++j) {
                    fprintf(fp, "h_binary[%d * %d + %d] = %f\n", i, size, j, h_binary[i * size + j]);
                }
            }
            fclose(fp);
        }
    }

    free(h_weights);
    free(h_binary);

    return 0;
}