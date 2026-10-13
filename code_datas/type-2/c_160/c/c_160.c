#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void upsample(float* in, int w, int h, int c, int batch, int stride, int forward, float scale, float* out) {
    clock_t start = clock();

    int i, j, k, b;

    for (int iter = 0; iter < 1000; iter++) {
        for (b = 0; b < batch; ++b) {
            for (k = 0; k < c; ++k) {
                for (j = 0; j < h * stride; ++j) {
                    for (i = 0; i < w * stride; ++i) {
                        int in_index = b * w * h * c + k * w * h + (j / stride) * w + i / stride;
                        int out_index = b * w * h * c * stride * stride + k * w * h * stride * stride + j * w * stride + i;

                        if (forward)
                            out[out_index] = scale * in[in_index];
                        else
                            in[in_index] += scale * out[out_index];
                    }
                }
            }
        }
    }

    double time_taken = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);
}

int main(int argc, char *argv[]) {
    int w = 128;
    int h = 128;
    int c = 64;
    int batch = 4;
    int stride = 2;
    int forward = 1; 
    float scale = 2.0;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* in = (float*)malloc(batch * w * h * c * sizeof(float));
    float* out = (float*)malloc(batch * w * h * c * stride * stride * sizeof(float));

    for (int i = 0; i < batch * w * h * c; i++) {
        in[i] = i + 1;
    }

    upsample(in, w, h, c, batch, stride, forward, scale, out);

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_160_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            int output_size = batch * w * h * c * stride * stride;
            for (int i = 0; i < output_size; i++) {
                fprintf(fp, "%.2f ", out[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }
    
    free(in);
    free(out);
    return 0;
}
