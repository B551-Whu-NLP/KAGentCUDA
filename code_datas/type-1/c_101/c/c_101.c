#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void grayscale(unsigned char* input, unsigned char* output, int size) {
    unsigned char r, g, b;
    clock_t start = clock();

    for (int iter = 0; iter < 100000; iter++) {
        for (int i = 0; i < size; i++) {
            r = input[3 * i];
            g = input[3 * i + 1];
            b = input[3 * i + 2];
            output[i] = (unsigned char)(0.21 * (float)r + 0.71 * (float)g + 0.07 * (float)b);
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);
}

int main(int argc, char* argv[]) {
    int size = 1024 * 100;
    unsigned char* h_input = (unsigned char*)malloc(3 * size * sizeof(unsigned char));
    unsigned char* h_output = (unsigned char*)malloc(size * sizeof(unsigned char));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < size * 3; ++i) {
        h_input[i] = i % 256;
    }

    grayscale(h_input, h_output, size);

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_101_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 100; i++) {
                fprintf(file, "%d ", h_output[i]);
            }
            fprintf(file, "\n");
            fclose(file);
        }
    }

    free(h_input);
    free(h_output);
    return 0;
}