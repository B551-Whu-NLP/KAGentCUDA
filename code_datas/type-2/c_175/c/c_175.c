#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void eltwise(int batch, int w1, int h1, int c1, float* add, int w2, int h2, int c2, float* out, int sum, int mult) {
    int i, j, k, b;
    int stride = w1 / w2;
    int sample = w2 / w1;

    if (stride < 1) stride = 1;
    if (sample < 1) sample = 1;

    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    if (mult == 1) {
        for (b = 0; b < batch; ++b) {
            for (k = 0; k < minc; ++k) {
                for (j = 0; j < minh; ++j) {
                    for (i = 0; i < minw; ++i) {
                        int out_index = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
                        int add_index = i * stride + w1 * (j * stride + h1 * (k + c1 * b));
                        out[out_index] = out[out_index] * add[add_index];
                    }
                }
            }
        }
    }
    else if (sum == 1) {
        for (b = 0; b < batch; ++b) {
            for (k = 0; k < minc; ++k) {
                for (j = 0; j < minh; ++j) {
                    for (i = 0; i < minw; ++i) {
                        int out_index = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
                        int add_index = i * stride + w1 * (j * stride + h1 * (k + c1 * b));
                        out[out_index] = out[out_index] + add[add_index];
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    const int batch = 10;
    const int w1 = 128, h1 = 144, c1 = 64;
    const int w2 = 64, h2 = 72, c2 = 32;
    const int sum = 1, mult = 0;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float *add = (float*)malloc(batch * w1 * h1 * c1 * sizeof(float));
    float *out = (float*)malloc(batch * w2 * h2 * c2 * sizeof(float));

    for (int i = 0; i < batch * w1 * h1 * c1; ++i) {
        add[i] = i + 1;
    }
    for (int i = 0; i < batch * w2 * h2 * c2; ++i) {
        out[i] = i + 1;
    }

    clock_t start = clock();
    eltwise(batch, w1, h1, c1, add, w2, h2, c2, out, sum, mult);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_175_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 100; ++i) {
                fprintf(fp, "%.2f ", out[i]);
            }
            fclose(fp);
        }
    }

    free(add);
    free(out);

    return 0;
}