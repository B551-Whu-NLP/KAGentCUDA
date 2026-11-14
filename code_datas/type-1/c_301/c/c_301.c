#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void normalize(float* x, float* mean, float* variance, int batch, int filters, int spatial) {
    for (int b = 0; b < batch; ++b) {
        for (int f = 0; f < filters; ++f) {
            for (int i = 0; i < spatial; ++i) {
                int index = b * filters * spatial + f * spatial + i;
                x[index] = (x[index] - mean[f]) / (sqrtf(variance[f]) + 0.000001f);
            }
        }
    }
}

int main(int argc, char** argv) {
    int batch = 10;
    int filters = 20;
    int spatial = 30;
    float* x = (float*)malloc(batch * filters * spatial * sizeof(float));
    float* mean = (float*)malloc(filters * sizeof(float));
    float* variance = (float*)malloc(filters * sizeof(float));
    
    for (int i = 0; i < batch * filters * spatial; i++) {
        x[i] = (float)(i + 1);
    }
    for (int i = 0; i < filters; i++) {
        mean[i] = (float)(2 * (i + 1));
    }
    for (int i = 0; i < filters; i++) {
        variance[i] = (float)(i + 1);
    }

    clock_t start = clock();

    normalize(x, mean, variance, batch, filters, spatial);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_301_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int b = 0; b < batch; ++b) {
                for (int f = 0; f < filters; ++f) {
                    for (int i = 0; i < spatial; ++i) {
                        int index = b * filters * spatial + f * spatial + i;
                        fprintf(file, "%.4f ", x[index]);
                    }
                    fprintf(file, "\n");
                }
            }
            fclose(file);
        }
    }

    free(x);
    free(mean);
    free(variance);
    return 0;
}