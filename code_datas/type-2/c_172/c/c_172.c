#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void col2im(const int n, const float* data_col, const int height, const int width, const int ksize,
    const int pad, const int stride, const int height_col, const int width_col,
    float* data_im) {

    for (int index = 0; index < n; index++) {
        float val = 0.0f;
        int w = index % width + pad;
        int h = (index / width) % height + pad;
        int c = index / (width * height);

        int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
        int w_col_end = (w / stride + 1 < width_col) ? w / stride + 1 : width_col;
        int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
        int h_col_end = (h / stride + 1 < height_col) ? h / stride + 1 : height_col;

        int offset = (c * ksize * ksize) * height_col * width_col + h * width_col + w;

        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                val += data_col[offset + h_col * width_col + w_col];
            }
        }

        data_im[index] += val;
    }
}

int main(int argc, char** argv) {
    const int channels = 16;
    const int height = 256;
    const int width = 256;
    const int ksize = 7;
    const int stride = 2;
    const int pad = 3;

    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int n = channels * height * width;

    float* data_col = (float*)malloc(channels * ksize * ksize * height_col * width_col * sizeof(float));
    float *data_im = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < channels * ksize * ksize * height_col * width_col; ++i) {
        data_col[i] = 1.0f;
    }

    for (int i = 0; i < n; ++i) {
        data_im[i] = 0.0f;
    }

    clock_t start = clock();
    col2im(n, data_col, height, width, ksize, pad, stride, height_col, width_col, data_im);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_172_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        int index = (c * height + h) * width + w;
                        fprintf(fp, "%.2f ", data_im[index]);
                    }
                    fprintf(fp, "\n");
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(data_col);
    free(data_im);

    return 0;
}