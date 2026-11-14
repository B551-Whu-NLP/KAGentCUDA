#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void convolutionRow(float *output, float *input, float *filter, int imageW, int imageH, int filterR) {
    int filterSize = 2 * filterR + 1;
    for (int i = 0; i < imageH; i++) {
        for (int j = 0; j < imageW; j++) {
            float sum = 0.0;
            for (int k = 0; k < filterSize; k++) {
                int idx = j - filterR + k;
                if (idx >= 0 && idx < imageW) {
                    sum += input[i * imageW + idx] * filter[k];
                }
            }
            output[i * imageW + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int imageW = 1000;
    int imageH = 1000;
    int filterR = 5;

    float *input = (float *)malloc(imageW * imageH * sizeof(float));
    float *filter = (float *)malloc((2 * filterR + 1) * sizeof(float));
    float *output = (float *)malloc(imageW * imageH * sizeof(float));

    for (int i = 0; i < imageW * imageH; i++) {
        input[i] = (i + 1) % 10;
    }

    for (int i = 0; i < 2 * filterR + 1; i++) {
        filter[i] = 0.1;
    }

    clock_t start = clock();

    convolutionRow(output, input, filter, imageW, imageH, filterR);

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE *fp = fopen("c_323_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < imageH; i++) {
                for (int j = 0; j < imageW; j++) {
                    fprintf(fp, "%.2f ", output[i * imageW + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(input);
    free(filter);
    free(output);

    return 0;
}