#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void nlf_filter_up_backward(const int n, const float* bottom_data, const float* top_data, const float* temp_diff, const int channel, const int height, const int width, const int wsize, float* filters_diff) {
    clock_t start = clock();

    for (int index = 0; index < n; index++) {
        int step = height * width;
        int base = index / step * step * channel + index % step;
        int fbase = index / step * step * wsize + index % step;
        int row = index % step / width;
        int col = index % step % width;

        for (int i = 0; i < channel; i++) {
            filters_diff[fbase] += temp_diff[base + i * step] * bottom_data[base + i * step];

            if (row + 1 < height)
                filters_diff[fbase + step] += temp_diff[base + i * step] * top_data[base + width + i * step];
            else
                filters_diff[fbase + step] += temp_diff[base + i * step] * bottom_data[base + i * step];

            if (row + 1 < height && col - 1 >= 0)
                filters_diff[fbase + 2 * step] += temp_diff[base + i * step] * top_data[base + width - 1 + i * step];
            else
                filters_diff[fbase + 2 * step] += temp_diff[base + i * step] * bottom_data[base + i * step];

            if (row + 1 < height && col + 1 < width)
                filters_diff[fbase + 3 * step] += temp_diff[base + i * step] * top_data[base + width + 1 + i * step];
            else
                filters_diff[fbase + 3 * step] += temp_diff[base + i * step] * bottom_data[base + i * step];

            if (col + 1 < width)
                filters_diff[fbase + 4 * step] += temp_diff[base + i * step] * top_data[base + 1 + i * step];
            else
                filters_diff[fbase + 4 * step] += temp_diff[base + i * step] * bottom_data[base + i * step];
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);
}

int main(int argc, char* argv[]) {
    int n = 100;
    int channel = 3;
    int height = 10;
    int width = 10;
    int wsize = 5;

    float* bottom_data_host = (float*)malloc(n * channel * height * width * sizeof(float));
    float* top_data_host = (float*)malloc(n * channel * height * width * sizeof(float));
    float* temp_diff_host = (float*)malloc(n * channel * height * width * sizeof(float));
    float* filters_diff_host = (float*)malloc(n * channel * wsize * sizeof(float));

    for (int i = 0; i < n * channel * height * width; i++) {
        bottom_data_host[i] = (float)(i % 100);
        top_data_host[i] = (float)(i % 100);
        temp_diff_host[i] = (float)(i % 100);
    }

    for (int i = 0; i < n * channel * wsize; i++) {
        filters_diff_host[i] = 0.0f;
    }

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    nlf_filter_up_backward(n, bottom_data_host, top_data_host, temp_diff_host, channel, height, width, wsize, filters_diff_host);

    if (verbose) {
        FILE* file = fopen("c_45_output.txt", "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.6f\n", filters_diff_host[i]);
            }
            fclose(file);
        }
    }

    free(bottom_data_host);
    free(top_data_host);
    free(temp_diff_host);
    free(filters_diff_host);
    return 0;
}