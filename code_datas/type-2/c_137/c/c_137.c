#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void variance(float* x, float* mean, int batch, int filters, int spatial, float* variance) {
    float scale = 1.f / (batch * spatial - 1);
    int i, j, k;

    for (i = 0; i < filters; ++i) {
        variance[i] = 0;

        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j * filters * spatial + i * spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }

        variance[i] *= scale;
    }
}

int main(int argc, char** argv) {
    int batch = 256;
    int filters = 512;
    int spatial = 2048;

    float* h_x = (float*)malloc(batch * filters * spatial * sizeof(float));
    float* h_mean = (float*)malloc(filters * sizeof(float));
    float* h_variance = (float*)malloc(filters * sizeof(float));

    for (int i = 0; i < batch * filters * spatial; ++i) {
        h_x[i] = 2.0f;
    }
    for (int i = 0; i < filters; ++i) {
        h_mean[i] = 1.0f;
    }

    clock_t start = clock();

    variance(h_x, h_mean, batch, filters, spatial, h_variance);

    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[] = "c_137_output.txt";
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "Filter %d: %f\n", i, h_variance[i]);
            }
            fclose(file);
        }
    }

    free(h_x);
    free(h_mean);
    free(h_variance);
    return 0;
}