#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void convolution(float* input, const float* mask, float* output, int array_size, int mask_size, int verbose) {
    int MASK_RADIUS = mask_size / 2;
    float temp = 0.0f;
    int ELEMENT_INDEX = 0;

    clock_t start = clock();

    for (int i = 0; i < array_size; i++) {
        temp = 0;

        for (int j = 0; j < mask_size; j++) {
            ELEMENT_INDEX = i - MASK_RADIUS + j;

            if (!(ELEMENT_INDEX < 0 || ELEMENT_INDEX >= array_size)) {
                temp += input[ELEMENT_INDEX] * mask[j];
            }
        }

        output[i] = temp;
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_128_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "%.6f ", output[i]);
            }
            fclose(file);
        }
    }
}

int main(int argc, char** argv) {
    int array_size = 1000000;
    int mask_size = 201;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* h_input = (float*)malloc(array_size * sizeof(float));
    float* h_mask = (float*)malloc(mask_size * sizeof(float));
    float* h_output = (float*)malloc(array_size * sizeof(float));

    srand(42);

    for (int i = 0; i < array_size; i++) {
        h_input[i] = (float)(rand() % 100) / 10.0f;
    }

    for (int i = 0; i < mask_size; i++) {
        h_mask[i] = 0.2f;
    }

    convolution(h_input, h_mask, h_output, array_size, mask_size, verbose);


    free(h_input);
    free(h_mask);
    free(h_output);
    return 0;
}