#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void nlf_filter_right_backward(const int n, const float* bottom_data, const float* top_data, const float* temp_diff, const int channel, const int height, const int width, const int wsize, float* filters_diff) {
    clock_t start = clock();

    for (int index = 0; index < n; index++) {
        int step = height * width;
        int base = index / step * step * channel + index % step;
        int fbase = index / step * step * wsize + index % step;
        int row = index % step / width;
        int col = index % step % width;

        for (int i = 0; i < channel; i++) {
            filters_diff[fbase] += temp_diff[base + i * step] * bottom_data[base + i * step];

            if (col - 1 >= 0)
                filters_diff[fbase + step] += temp_diff[base + i * step] * top_data[base - 1 + i * step];
            else
                filters_diff[fbase + step] += temp_diff[base + i * step] * bottom_data[base + i * step];

            if (col - 1 >= 0 && row - 1 >= 0)
                filters_diff[fbase + 2 * step] += temp_diff[base + i * step] * top_data[base - width - 1 + i * step];
            else
                filters_diff[fbase + 2 * step] += temp_diff[base + i * step] * bottom_data[base + i * step];

            if (col - 1 >= 0 && row + 1 < height)
                filters_diff[fbase + 3 * step] += temp_diff[base + i * step] * top_data[base + width - 1 + i * step];
            else
                filters_diff[fbase + 3 * step] += temp_diff[base + i * step] * bottom_data[base + i * step];

            if (row - 1 >= 0)
                filters_diff[fbase + 4 * step] += temp_diff[base + i * step] * top_data[base - width + i * step];
            else
                filters_diff[fbase + 4 * step] += temp_diff[base + i * step] * bottom_data[base + i * step];
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);
}

int main(int argc, char *argv[]) {
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    int n = 10000;
    int channel = 10;
    int height = 50;
    int width = 50;
    int wsize = 5;

    float* bottom_data_host = (float*)malloc(n * channel * height * width * sizeof(float));
    float* top_data_host = (float*)malloc(n * channel * height * width * sizeof(float));
    float* temp_diff_host = (float*)malloc(n * channel * height * width * sizeof(float));
    float* filters_diff_host = (float*)malloc(n * channel * wsize * sizeof(float));

    srand(0);
    for (int i = 0; i < n * channel * height * width; i++) {
        bottom_data_host[i] = (float)(rand() % 100);
        top_data_host[i] = (float)(rand() % 100);
        temp_diff_host[i] = (float)(rand() % 100);
    }

    for (int i = 0; i < n * channel * wsize; i++) {
        filters_diff_host[i] = 0.0f;
    }

    nlf_filter_right_backward(n, bottom_data_host, top_data_host, temp_diff_host, channel, height, width, wsize, filters_diff_host);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_122_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.6f\n", filters_diff_host[i]);
            }
            fclose(fp);
        }
    }

    free(bottom_data_host);
    free(top_data_host);
    free(temp_diff_host);
    free(filters_diff_host);

    return 0;
}