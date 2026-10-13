#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void nlf_right_forward(const int n, const float* filters, const int channel, const int height, const int width, const int wsize, float* top_data) {
    clock_t start = clock();

    for (int index = 0; index < n; index++) {
        int step = height * width;
        int base = index * step;
        int fbase = index / channel * wsize * step;

        for (int col = 0; col < width; col++) {
            for (int row = 0; row < height; row++) {
                float temp = 0;
                int r, c, shift;

                r = row;
                c = col;
                shift = 0 * step + row * width + col;
                temp += top_data[base + r * width + c] * filters[fbase + shift];

                r = row;
                c = col - 1;
                shift = 1 * step + row * width + col;
                if (c >= 0)
                    temp += top_data[base + r * width + c] * filters[fbase + shift];
                else
                    temp += top_data[base + row * width + col] * filters[fbase + shift];

                r = row - 1;
                c = col - 1;
                shift = 2 * step + row * width + col;
                if (c >= 0 && r >= 0)
                    temp += top_data[base + r * width + c] * filters[fbase + shift];
                else
                    temp += top_data[base + row * width + col] * filters[fbase + shift];

                r = row + 1;
                c = col - 1;
                shift = 3 * step + row * width + col;
                if (c >= 0 && r < height)
                    temp += top_data[base + r * width + c] * filters[fbase + shift];
                else
                    temp += top_data[base + row * width + col] * filters[fbase + shift];

                r = row - 1;
                c = col;
                shift = 4 * step + row * width + col;
                if (r >= 0)
                    temp += top_data[base + r * width + c] * filters[fbase + shift];
                else
                    temp += top_data[base + row * width + col] * filters[fbase + shift];

                top_data[base + row * width + col] = temp;
            }
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);
}

int main(int argc, char *argv[]) {
    int n = 100;
    int channel = 30;
    int height = 128;
    int width = 128;
    int wsize = 5;

    float* filters_host = (float*)malloc(n * channel * wsize * height * width * sizeof(float));
    float* top_data_host = (float*)malloc(n * height * width * sizeof(float));

    for (int i = 0; i < n * channel * wsize * height * width; i++) {
        filters_host[i] = (float)(i % 10);
    }
    for (int i = 0; i < n * height * width; i++) {
        top_data_host[i] = (float)(i % 5);
    }

    nlf_right_forward(n, filters_host, channel, height, width, wsize, top_data_host);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_159_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < height * width; j++) {
                    fprintf(fp, "top_data[%d][%d] = %f\n", i, j, top_data_host[i * height * width + j]);
                }
            }
            fclose(fp);
        }
    }

    free(filters_host);
    free(top_data_host);

    return 0;
}