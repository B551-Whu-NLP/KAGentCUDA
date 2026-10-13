#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void conv1x1(int input_channels, int input_size, int n, float* input_im, float* filter_weight, float* filter_bias, float* output_im) {
    clock_t start = clock();

    for (int filter_index = 0; filter_index < n; filter_index++) {
        float* current_filter_weight = filter_weight + filter_index * input_channels;
        float bias = filter_bias[filter_index];
        float* current_output_im = output_im + filter_index * input_size * input_size;

        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < input_size; j++) {
                float tmp = bias;
                for (int k = 0; k < input_channels; k++) {
                    tmp += input_im[k * input_size * input_size + i * input_size + j] * current_filter_weight[k];
                }
                current_output_im[i * input_size + j] = (tmp > 0.0) ? tmp : 0.0;
            }
        }
    }

    clock_t end = clock();
    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_spent * 1000);
}

int main(int argc, char *argv[]) {
    int input_channels = 3;
    int input_size = 5;
    int n = 8;
    int total_elements = n * input_size * input_size;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* h_input_im = (float*)malloc(input_channels * input_size * input_size * sizeof(float));
    float* h_filter_weight = (float*)malloc(n * input_channels * sizeof(float));
    float* h_filter_bias = (float*)malloc(n * sizeof(float));
    float* h_output_im = (float*)malloc(total_elements * sizeof(float));

    for (int i = 0; i < input_channels * input_size * input_size; ++i) {
        h_input_im[i] = (float)(i % 256);
    }
    for (int i = 0; i < n * input_channels; ++i) {
        h_filter_weight[i] = (float)(i % 256) / 256.0f;
    }
    for (int i = 0; i < n; ++i) {
        h_filter_bias[i] = (float)(i % 10);
    }

    conv1x1(input_channels, input_size, n, h_input_im, h_filter_weight, h_filter_bias, h_output_im);

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_336_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int f = 0; f < n; f++) {
                for (int i = 0; i < input_size; i++) {
                    for (int j = 0; j < input_size; j++) {
                        fprintf(fp, "%.2f ", h_output_im[f * input_size * input_size + i * input_size + j]);
                    }
                    fprintf(fp, "\n");
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_input_im);
    free(h_filter_weight);
    free(h_filter_bias);
    free(h_output_im);

    return 0;
}