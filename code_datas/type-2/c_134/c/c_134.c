#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void Column(const float* filter, const float* buffer, float* output, int imageW, int imageH, int filterR) {
    int x, y, k;
    for (y = 0; y < imageH; y++) {
        for (x = 0; x < imageW; x++) {
            float sum = 0;
            for (k = -filterR; k <= filterR; k++) {
                int d = y + k;
                if (d >= 0 && d < imageH) {
                    sum += buffer[d * imageW + x] * filter[filterR - k];
                }
            }
            output[y * imageW + x] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int imageW = 2048;
    int imageH = 2048;
    int filterR = 63;

    float* h_filter = (float*)malloc((2 * filterR + 1) * sizeof(float));
    float* h_buffer = (float*)malloc(imageW * imageH * sizeof(float));
    float* h_output = (float*)malloc(imageW * imageH * sizeof(float));

    for (int i = 0; i < (2 * filterR + 1); i++) {
        h_filter[i] = 1.0f / (2 * filterR + 1);
    }

    for (int i = 0; i < imageW * imageH; i++) {
        h_buffer[i] = i % 256;
    }

    clock_t start = clock();
    Column(h_filter, h_buffer, h_output, imageW, imageH, filterR);
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_134_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                for (int j = 0; j < 1000; j++) {
                    fprintf(fp, "%.2f ", h_output[i * imageW + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }


    free(h_filter);
    free(h_buffer);
    free(h_output);
    return 0;
}