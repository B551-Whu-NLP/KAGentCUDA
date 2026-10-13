#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void forward_avgpool_layer(int n, int w, int h, int c, float* input, float* output) {
    int b, i, k;
    for (b = 0; b < n; ++b) {
        for (k = 0; k < c; ++k) {
            int out_index = k + b * c;
            output[out_index] = 0;
            for (i = 0; i < h * w; ++i) {
                int in_index = i + h * w * (k + b * c);
                output[out_index] += input[in_index];
            }
            output[out_index] /= h * w;
        }
    }
}

int main(int argc, char** argv) {
    int n = 100;
    int w = 128;
    int h = 128;
    int c = 64;

    float* h_input = (float*)malloc(n * w * h * c * sizeof(float));
    float* h_output = (float*)malloc(n * c * sizeof(float));

    for (int i = 0; i < n * w * h * c; ++i) {
        h_input[i] = i % 10;
    }

    clock_t start = clock();
    forward_avgpool_layer(n, w, h, c, h_input, h_output);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_133_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%f ", h_output[i]);
            }
            fprintf(file, "\n");
            fclose(file);
        }
    }

    free(h_input);
    free(h_output);
    return 0;
}