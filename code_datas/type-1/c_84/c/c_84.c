#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void convertEdgeMaskToFloat(float* d_output, unsigned char* d_input, unsigned int width, unsigned int height, double* time_spent) {
    clock_t start = clock();

    for (int iter = 0; iter < 1000; iter++) {
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int index = y * width + x;
                d_output[index] = (float)d_input[index];
            }
        }
    }

    clock_t end = clock();
    *time_spent = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
}

int main(int argc, char* argv[]) {
    unsigned int width = 2048;
    unsigned int height = 2048;
    float* h_output = (float*)malloc(width * height * sizeof(float));
    unsigned char* h_input = (unsigned char*)malloc(2 * width * height * sizeof(unsigned char));
    
    srand(42);
    
    for (unsigned int i = 0; i < width * height; ++i) {
        h_input[i] = (unsigned char)(rand() % 256);
    }

    double time_spent;
    convertEdgeMaskToFloat(h_output, h_input, width, height, &time_spent);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_84_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (unsigned int i = 0; i < 1000; ++i) {
                for (unsigned int j = 0; j < 1000; ++j) {
                    fprintf(file, "%4.2f ", h_output[i * width + j]);
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    printf("Time: %.2f(ms)\n", time_spent);
    free(h_output);
    free(h_input);

    return 0;
}