#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void mean_kernel(float* x, int batch, int filters, int spatial, float* mean) {
    float scale = 1.0 / (batch * spatial);
    int i, j, k;

    clock_t start = clock();

    for (i = 0; i < filters; ++i) {
        mean[i] = 0;

        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j * filters * spatial + i * spatial + k;
                mean[i] += x[index];
            }
        }

        mean[i] *= scale;
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);
}

int main(int argc, char *argv[]) {
    int batch = 16;
    int filters = 32;
    int spatial = 64;

    float *x = (float *)malloc(batch * filters * spatial * sizeof(float));
    float *mean = (float *)malloc(filters * sizeof(float));

    for (int i = 0; i < batch * filters * spatial; i++) {
        x[i] = (float)(i % 100 + 1);
    }

    mean_kernel(x, batch, filters, spatial, mean);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE *fp = fopen("c_318_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < filters; i++) {
                fprintf(fp, "%.6f ", mean[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }

    free(x);
    free(mean);

    return 0;
}