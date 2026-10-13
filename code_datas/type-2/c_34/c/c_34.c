#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void nlf_left_forward(const int n, const float* filters, const int channel, const int height, const int width, const int wsize, float* top_data) {
    clock_t start = clock();

    for (int index = 0; index < n; index++) {
        int step = height * width;
        int base = index * step;
        int fbase = index * channel * wsize * step;

        for (int col = width - 1; col >= 0; col--) {
            for (int row = height - 1; row >= 0; row--) {
                float temp = 0;
                int r, c, shift;

                r = row;
                c = col;
                shift = 0 * step + row * width + col;
                temp += top_data[base + r * width + c] * filters[fbase + shift];

                r = row;
                c = col + 1;
                shift = 1 * step + row * width + col;
                if (c < width)
                    temp += top_data[base + r * width + c] * filters[fbase + shift];
                else
                    temp += top_data[base + row * width + col] * filters[fbase + shift];

                r = row - 1;
                c = col + 1;
                shift = 2 * step + row * width + col;
                if (c < width && r >= 0)
                    temp += top_data[base + r * width + c] * filters[fbase + shift];
                else
                    temp += top_data[base + row * width + col] * filters[fbase + shift];

                r = row + 1;
                c = col + 1;
                shift = 3 * step + row * width + col;
                if (c < width && r < height)
                    temp += top_data[base + r * width + c] * filters[fbase + shift];
                else
                    temp += top_data[base + row * width + col] * filters[fbase + shift];

                r = row + 1;
                c = col;
                shift = 4 * step + row * width + col;
                if (r < height)
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

int main(int argc, char** argv) {
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    int n = 100;
    int channel = 3;
    int height = 64;
    int width = 64;
    int wsize = 5;

    float* filters_host = (float*)malloc(n * channel * wsize * height * width * sizeof(float));
    float* top_data_host = (float*)malloc(n * height * width * sizeof(float));

    srand(0);
    for (int i = 0; i < n * channel * wsize * height * width; i++) {
        filters_host[i] = (float)(rand() % 100) / 100.0f;
    }

    for (int i = 0; i < n * height * width; i++) {
        top_data_host[i] = (float)(rand() % 100) / 100.0f;
    }

    nlf_left_forward(n, filters_host, channel, height, width, wsize, top_data_host);

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_34_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "top_data_host[%d] = %f\n", i, top_data_host[i]);
            }
            fclose(file);
        }
    }

    free(filters_host);
    free(top_data_host);

    return 0;
}