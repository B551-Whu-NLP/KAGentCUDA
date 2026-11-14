#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

void reorg(float* x, int w, int h, int c, int batch, int stride, int forward, float* out) {
    int b, i, j, k;
    int out_c = c / (stride * stride);
    for (b = 0; b < batch; ++b) {
        for (k = 0; k < c; ++k) {
            for (j = 0; j < h; ++j) {
                for (i = 0; i < w; ++i) {
                    int in_index = i + w * (j + h * (k + c * b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i * stride + offset % stride;
                    int h2 = j * stride + offset / stride;
                    int out_index = w2 + w * stride * (h2 + h * stride * (c2 + out_c * b));
                    if (forward)
                        out[out_index] = x[in_index];
                    else
                        out[in_index] = x[out_index];
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    const int w = 512;
    const int h = 512;
    const int c = 512;
    const int batch = 8;
    const int stride = 2;
    const int out_c = c / (stride * stride);
    const int out_w = w * stride;
    const int out_h = h * stride;

    size_t in_size  = batch * c * h * w * sizeof(float);
    size_t out_size = batch * out_c * out_h * out_w * sizeof(float);

    float* x   = (float*)malloc(in_size);
    float* out = (float*)malloc(out_size);

    for (int i = 0; i < batch * c * h * w; ++i) {
        x[i] = (float)(i % 100);
    }

    clock_t start = clock();
    reorg(x, w, h, c, batch, stride, 1, out);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_343_output.txt");
        FILE* f = fopen(output_file, "w");
        if (f != NULL) {
            for (int i = 0; i < batch * out_c; ++i) {
                fprintf(f, "out[%d] = %.2f\n", i, out[i]);
            }
            fclose(f);
        }
    }

    free(x);
    free(out);
    return 0;
}