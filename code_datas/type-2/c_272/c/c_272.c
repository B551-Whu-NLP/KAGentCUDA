#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void reduce(float* input, float* output, int size) {
    float sum = 0.0;

    for (int i = 0; i < size; i++) {
        sum += input[i];
    }

    output[0] = sum;
}

int main(int argc, char* argv[]) {
    int size = 100000000; // 增大数组大小
    float* h_input = (float*)malloc(size * sizeof(float));
    float* h_output = (float*)malloc(size * sizeof(float));

    for (int i = 0; i < size; ++i) {
        h_input[i] = 1.0f;
    }

    clock_t start = clock();
    reduce(h_input, h_output, size);
    clock_t end = clock();

    double time_taken = (double)(end - start) * 1000 / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_taken);
    printf("Final Result: %.2f\n", h_output[0]);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_272_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            fprintf(file, "Final Result: %.2f\n", h_output[0]);
            fclose(file);
        }
    }

    free(h_input);
    free(h_output);

    return 0;
}