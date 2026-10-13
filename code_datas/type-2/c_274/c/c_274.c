#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void avgpool(int n, float* input_im, float* output_im) {
    clock_t start = clock();

    for (int class_index = 0; class_index < n; class_index++) {
        float* tmp_input = input_im + 169 * class_index;
        float tmp = 0.0f;

        for (int i = 0; i < 169; i++) {
            tmp += tmp_input[i];
        }

        output_im[class_index] = tmp / 169.0;
    }

    clock_t end = clock();
    double time_spent = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_spent);
}

int main(int argc, char** argv) {
    int num_classes = 100000; // 进一步增加类数量
    float* h_input_im = (float*)malloc(169 * num_classes * sizeof(float));
    float* h_output_im = (float*)malloc(num_classes * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    srand(0);

    for (int i = 0; i < 169 * num_classes; ++i) {
        h_input_im[i] = (float)(rand() % 100);
    }

    avgpool(num_classes, h_input_im, h_output_im);

    if (verbose) {
        FILE* file = fopen("c_274_output.txt", "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%.2f ", h_output_im[i]);
            }
            fclose(file);
        }
    }

    free(h_input_im);
    free(h_output_im);

    return 0;
}