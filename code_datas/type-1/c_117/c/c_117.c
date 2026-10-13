#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void remove_layer(float* output, float* input, unsigned int width, unsigned int height, unsigned int depth) {
    for (unsigned int channel = 0; channel < depth; channel++)
        for (unsigned int row = 0; row < height; row++)
            for (unsigned int column = 0; column < (width - 1); column++) {
                unsigned int idx = (channel * height + row) * width + column;
                output[idx] = input[idx] - input[idx + 1];
            }
}

int main(int argc, char* argv[]) {
    unsigned int width = 1000;
    unsigned int height = 1000;
    unsigned int depth = 30;

    float* h_output = (float*)malloc(width * height * depth * sizeof(float));
    float* h_input = (float*)malloc(width * height * depth * sizeof(float));

    for (unsigned int i = 0; i < width * height * depth; i++) {
        h_input[i] = (float)(i % 255);
    }
    for (unsigned int i = 0; i < width * height * depth; i++) {
        h_output[i] = (float)0;
    }

    clock_t start = clock();
    remove_layer(h_output, h_input, width, height, depth);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;

    printf("Time: %.2f(ms)\n", time_taken);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        FILE* fp = fopen("c_117_output.txt", "w");
        if (fp != NULL) {
            for (unsigned int i = 0; i < 1000; i++) {
                fprintf(fp, "Output[%d] = %f\n", i, h_output[i]);
            }
            fclose(fp);
        }
    }

    free(h_output);
    free(h_input);
    return 0;
}